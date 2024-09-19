# utils/fisher_regularization.py

import torch

class FisherRegularization(object):
    def __init__(self, model, device, coefficient=0.001):
        self.model = model
        self.device = device
        self.coefficient = coefficient
        self.fisher_info = {}
        self.previous_params = {}

    def compute_fisher(self, data_loader):
        self.model.eval()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_info[name] = torch.zeros_like(param.data).to(self.device)
                self.previous_params[name] = param.data.clone().to(self.device)

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.model.zero_grad()
            outputs, loss = self.model(input_ids=input_ids, targets=labels)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_info[name] += param.grad.data.pow(2)
        self.model.train()

    def compute_fisher_loss(self):
        fisher_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher = self.fisher_info[name]
                delta = param.data - self.previous_params[name]
                fisher_loss += (fisher * delta.pow(2)).sum()
        return self.coefficient * fisher_loss
