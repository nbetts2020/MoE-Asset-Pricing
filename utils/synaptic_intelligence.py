import torch

class SI(object):
    def __init__(self, model, device, coefficient=0.001):
        self.model = model
        self.device = device
        self.coefficient = coefficient
        self.importances = {}
        self.previous_params = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importances[name] = torch.zeros_like(param.data).to(self.device)
                self.previous_params[name] = param.data.clone().to(self.device)

    def update_importances(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                delta = param.data - self.previous_params[name]
                self.importances[name] += (param.grad * delta).detach().abs()
                self.previous_params[name] = param.data.clone()

    def compute_si_loss(self):
        si_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance = self.importances[name]
                delta = param.data - self.previous_params[name]
                si_loss += (importance * delta.pow(2)).sum()
        return self.coefficient * si_loss
