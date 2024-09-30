# utils/si.py

import torch
import copy

class SynapticIntelligence:
    def __init__(self, model, lambda_si=1.0):
        """
        Initialize Synaptic Intelligence.

        Args:
            model (nn.Module): The PyTorch model.
            lambda_si (float): The regularization strength.
        """
        self.model = model
        self.lambda_si = lambda_si
        self.omega = {}
        self.theta_star = {}
        self.previous_params = {}
        self.accumulated_gradients = {}
        self.total_path = 0.0
        self.epsilon = 1e-8  # to prevent division by zero

        # Initialize omega and theta_star
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param.data)
                self.theta_star[name] = param.data.clone()
                self.previous_params[name] = param.data.clone()
                self.accumulated_gradients[name] = torch.zeros_like(param.data)

    def update_omega(self, loss, batch_size):
        """
        Update omega based on parameter changes and gradients.

        Args:
            loss (torch.Tensor): The current loss.
            batch_size (int): The number of samples in the batch.
        """
        # Compute gradients w.r.t. parameters
        grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
        grads = list(grads)
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if param.requires_grad:
                delta_theta = param.data - self.previous_params[name]
                self.accumulated_gradients[name] += grad * delta_theta

        self.total_path += loss.item() * batch_size

    def consolidate_omega(self):
        """
        Consolidate omega after a task is completed.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] += (self.accumulated_gradients[name] / (self.total_path + self.epsilon)).abs()
                self.theta_star[name] = param.data.clone()
                self.previous_params[name] = param.data.clone()
                self.accumulated_gradients[name].zero_()
        self.total_path = 0.0

    def penalty(self):
        """
        Compute the SI penalty.

        Returns:
            torch.Tensor: The SI regularization term.
        """
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                penalty += (self.omega[name] * (param - self.theta_star[name]).pow(2)).sum()
        return self.lambda_si * penalty

    def save_state(self, filepath):
        """
        Save the SI state.

        Args:
            filepath (str): Path to save the SI state.
        """
        state = {
            'omega': {name: param.cpu() for name, param in self.omega.items()},
            'theta_star': {name: param.cpu() for name, param in self.theta_star.items()},
            'previous_params': {name: param.cpu() for name, param in self.previous_params.items()},
            'accumulated_gradients': {name: param.cpu() for name, param in self.accumulated_gradients.items()},
            'total_path': self.total_path
        }
        torch.save(state, filepath)

    def load_state(self, filepath):
        """
        Load the SI state.

        Args:
            filepath (str): Path to load the SI state from.
        """
        state = torch.load(filepath, map_location='cpu')
        for name in self.omega:
            if name in state['omega']:
                self.omega[name] = state['omega'][name].to(self.model.device)
            if name in state['theta_star']:
                self.theta_star[name] = state['theta_star'][name].to(self.model.device)
            if name in state['previous_params']:
                self.previous_params[name] = state['previous_params'][name].to(self.model.device)
            if name in state['accumulated_gradients']:
                self.accumulated_gradients[name] = state['accumulated_gradients'][name].to(self.model.device)
        self.total_path = state.get('total_path', 0.0)
