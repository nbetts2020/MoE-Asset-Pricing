import torch
import os
import argparse
from transformers import AutoTokenizer
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights
from utils.config import *

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run'], help="Mode: 'train' to train the model, 'run' to generate text")
    parser.add_argument('input_text', type=str, nargs='?', help="Input text to complete (required if mode is 'run')", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")

    args = parser.parse_args()

    torch.manual_seed(1337)

    # Detect the available device(s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"Using device: {device}, Number of GPUs: {n_gpus}")

    # Load the model with the tokenizer
    model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
    tokenizer = model.tokenizer

    if args.mode == 'train':
        model.apply(kaiming_init_weights)

        if n_gpus > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Load and tokenize the text data
        input_path = os.path.join("data", "TinyStoriesV2-GPT4-valid.txt")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

        train_model(model, optimizer, max_iters, eval_interval, device, tokens)
    
    elif args.mode == 'run':
        if not args.input_text:
            raise ValueError("You must provide input text when mode is 'run'")

        # Load the model weights
        directory = "model"
        filename = "model_weights.pth"
        filepath = os.path.join(directory, filename)

        model = model.to(device)
        model.load_state_dict(torch.load(filepath, map_location=device))

        if n_gpus > 1:
            model = torch.nn.DataParallel(model)

        model.eval()

        # Tokenize the input text
        encoded_input = tokenizer(args.input_text, return_tensors='pt').to(device)
        idx = encoded_input["input_ids"]
        print(idx)
        print(tokenizer.decode(idx[0]))

        # Generate continuation
        generated_tokens = model.generate(idx, max_new_tokens=100, stream=True)
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
