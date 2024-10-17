# utils/ewc.py

import torch
import copy
import torch.distributed as dist
from torch.cuda.amp.autocast_mode import autocast

class ElasticWeightConsolidation:
    def __init__(self, model, dataloader, device, args):
        """
        Initialize Elastic Weight Consolidation (EWC).

        Args:
            model (nn.Module): The model to apply EWC to.
            dataloader (DataLoader): DataLoader for computing Fisher Information.
            device (torch.device): Device to perform computations on.
            args (argparse.Namespace): Command-line arguments containing DDP usage.
        """
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.device = device
        self.args = args

        # Store model parameters after training on task
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        # Initialize Fisher Information matrix
        self.fisher = self._compute_fisher_information()

    def _compute_fisher_information(self):
        """
        Compute the Fisher Information matrix for the model parameters.

        Returns:
            dict: A dictionary mapping parameter names to their Fisher Information tensors.
        """
        # Initialize Fisher Information matrix
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()

        for batch in self.dataloader:
            self.model.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast(device_type='cuda'):
                outputs, _ = self.model(input_ids=input_ids)
                loss = torch.nn.functional.mse_loss(outputs.squeeze(), labels.float())

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += (p.grad.detach() ** 2)

        # Average Fisher Information over all batches
        for n in fisher:
            fisher[n] /= len(self.dataloader)

        # If using DDP and multiple GPUs, average Fisher across all processes
        if self.args.use_ddp and torch.cuda.device_count() > 1:
            for n in fisher:
                dist.all_reduce(fisher[n], op=dist.ReduceOp.SUM)
                fisher[n] /= dist.get_world_size()

        return fisher

    def penalty(self, model):
        """
        Compute the EWC penalty for the current model parameters.

        Args:
            model (nn.Module): The current model.

        Returns:
            torch.Tensor: The EWC penalty.
        """
        loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad:
                # Get parameter value before training on the new task
                mean = self.params[n]
                # Get Fisher Information for this parameter
                fisher = self.fisher[n]
                # Compute the EWC loss
                loss += (fisher * (p - mean) ** 2).sum()
        return loss

    def save_state(self, path):
        """
        Save the EWC state to a file.

        Args:
            path (str): Path to save the EWC state.
        """
        state = {
            'params': {n: p.cpu() for n, p in self.params.items()},
            'fisher': {n: f.cpu() for n, f in self.fisher.items()}
        }
        torch.save(state, path)

    def load_state(self, path):
        """
        Load the EWC state from a file.

        Args:
            path (str): Path to load the EWC state from.
        """
        state = torch.load(path, map_location=self.device)
        self.params = {n: p.to(self.device) for n, p in state['params'].items()}
        self.fisher = {n: f.to(self.device) for n, f in state['fisher'].items()}
