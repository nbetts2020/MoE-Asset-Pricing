# utils/si.py

import torch

class SynapticIntelligence:
    def __init__(self, model, lambda_si=1.0):
        self.model = model
        self.lambda_si = lambda_si
        self.omega = {}
        self.theta_star = {}
        self.previous_params = {}
        self.accumulated_gradients = {}
        self.total_loss = 0.0
        self.epsilon = 1e-8  # To prevent division by zero

        # Initialize omega and theta_star
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param.data)
                self.theta_star[name] = param.data.clone()
                self.previous_params[name] = param.data.clone()
                self.accumulated_gradients[name] = torch.zeros_like(param.data)

    def update_omega(self):
        """
        Update omega based on parameter changes and gradients.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta_theta = param.data - self.previous_params[name]
                if param.grad is not None:
                    self.accumulated_gradients[name] += param.grad * delta_theta
                self.previous_params[name] = param.data.clone()

    def consolidate_omega(self):
        """
        Consolidate omega after a task is completed.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] += (self.accumulated_gradients[name] / (self.total_loss + self.epsilon)).abs()
                self.theta_star[name] = param.data.clone()
                self.accumulated_gradients[name].zero_()
                self.previous_params[name] = param.data.clone()
        self.total_loss = 0.0

    def penalty(self):
        """
        Compute the SI penalty.
        """
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                penalty += (self.omega[name] * (param - self.theta_star[name]).pow(2)).sum()
        return self.lambda_si * penalty

    def save_state(self, filepath):
        """
        Save the SI state.
        """
        state = {
            'omega': {name: param.cpu() for name, param in self.omega.items()},
            'theta_star': {name: param.cpu() for name, param in self.theta_star.items()},
            'previous_params': {name: param.cpu() for name, param in self.previous_params.items()},
            'accumulated_gradients': {name: param.cpu() for name, param in self.accumulated_gradients.items()},
            'total_loss': self.total_loss
        }
        torch.save(state, filepath)

    def load_state(self, filepath):
        """
        Load the SI state.
        """
        state = torch.load(filepath, map_location='cpu')
        for name in self.omega:
            if name in state['omega']:
                device = next(self.model.parameters()).device
                self.omega[name] = state['omega'][name].to(device)
            if name in state['theta_star']:
                self.theta_star[name] = state['theta_star'][name].to(self.model.device)
            if name in state['previous_params']:
                self.previous_params[name] = state['previous_params'][name].to(self.model.device)
            if name in state['accumulated_gradients']:
                self.accumulated_gradients[name] = state['accumulated_gradients'][name].to(self.model.device)
        self.total_loss = state.get('total_loss', 0.0)
