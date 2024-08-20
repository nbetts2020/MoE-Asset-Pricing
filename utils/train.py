import torch
from utils.data import get_batch, estimate_loss
import os
from utils.config import *

def train_model(model, optimizer, max_iters, eval_interval, device, text, encoder):
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters=eval_iters, device=device, text=text, encoder=encoder)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', text, encoder)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    directory = "model"
    filename = "model_weights.pth"

    # Create the full path
    filepath = os.path.join(directory, filename)

    # Save the model weights
    torch.save(model.state_dict(), filepath)