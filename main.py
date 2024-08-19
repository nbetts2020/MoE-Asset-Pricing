# main.py

import torch
from utils.data import *
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights
import os
import argparse
from utils.config import *

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run'], help="Mode: 'train' to train the model, 'run' to generate text")
    parser.add_argument('input_text', type=str, nargs='?', help="Input text to complete (required if mode is 'run')", default=None)

    args = parser.parse_args()

    torch.manual_seed(1337)

    # Detect the available device(s)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpus = 0

    print(f"Using device: {device}, Number of GPUs: {n_gpus}")

    input_path = os.path.join("data", "shakespeare.txt")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))

    encode = encoder(chars)
    decode = decoder(chars)

    # Load model architecture
    model = SparseMoELanguageModel(text=text, encoder=encode, decoder=decode, chars=chars)

    if args.mode == 'train':
        model.apply(kaiming_init_weights)

        # Handle multi-GPU setup
        if n_gpus > 1:
            model = torch.nn.DataParallel(model)
        
        model = model.to(device)
        print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        # Start training
        train_model(model, optimizer, max_iters, eval_interval, device, text, encode)
    
    elif args.mode == 'run':
        if not args.input_text:
            raise ValueError("You must provide input text when mode is 'run'")

        # Define the directory and filename for the model weights
        directory = "model"
        filename = "model_weights.pth"
        filepath = os.path.join(directory, filename)

        model = model.to(device)

        # Load the model weights
        model.load_state_dict(torch.load(filepath, map_location=device))

        # Handle multi-GPU setup for inference
        if n_gpus > 1:
            model = torch.nn.DataParallel(model)

        # Set model to evaluation mode
        model.eval()

        # Encode input and generate continuation
        encode_list = encode(args.input_text)
        context = torch.tensor([encode_list], dtype=torch.long, device=device)
        generated_tensor = model.generate(context, decoder=decode, max_new_tokens=100, stream=True)[0]

        # Decode the generated tensor to text
        generated_text = decode(generated_tensor.tolist())
        
        print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
