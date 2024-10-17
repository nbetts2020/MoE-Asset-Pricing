# utils/ewc.py

import torch
import copy
from torch.amp.autocast_mode import autocast

class ElasticWeightConsolidation:
    def __init__(self, model, dataloader, device):
        self.model = copy.deepcopy(model)
        self.dataloader = dataloader
        self.device = device

        # Store model parameters after training on task
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        # Initialize Fisher Information matrix
        self.fisher = self._compute_fisher_information()

    def _compute_fisher_information(self):
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

        return fisher

    def penalty(self, model):
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
