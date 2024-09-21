import torch
import os
import argparse
from utils.model import SparseMoELanguageModel
from utils.train import train_model
from utils.utils import kaiming_init_weights, get_data, prepare_dataloader, load_model_weights, initialize_si
from utils.config import *
from torch.utils.data import DataLoader
from utils.data import ArticlePriceDataset
from utils.si import SynapticIntelligence
from utils.test import test_forgetting
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="SparseMoE Language Model")
    parser.add_argument('mode', choices=['train', 'run', 'test_forgetting'], help="Mode: 'train', 'run', or 'test_forgetting'")
    parser.add_argument('input_text', type=str, nargs='?', help="Input article text (required if mode is 'run' without --test)", default=None)
    parser.add_argument('--tokenizer_name', type=str, default="gpt2", help="Name of the pretrained tokenizer to use")
    parser.add_argument('--test', action='store_true', help="If specified in 'run' mode, evaluate the model on the test set.")
    parser.add_argument('--update', action='store_true', help="If specified in 'train' mode, update a pre-existing model with SI.")

    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and tokenizer
    model = SparseMoELanguageModel(tokenizer_name=args.tokenizer_name)
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    if args.mode == 'test_forgetting':
        # Load pre-trained model
        model = load_model_weights(model, 'model/model_weights.pth', device)

        # Prepare tasks
        task_dataloaders = prepare_tasks()

        # Initialize Synaptic Intelligence (SI) if update flag is specified
        si = initialize_si(model, 'model/si_state.pth', lambda_si=lambda_si) if args.update else None

        # Run catastrophic forgetting test
        results = test_forgetting(model, task_dataloaders, optimizer, EPOCHS, device, si=si)
        print(results)

    elif args.mode == 'train':
        model.apply(kaiming_init_weights)

        # Check for multiple GPUs
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.to(device)
        print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f} million parameters")

        # Initialize optimizer
        param_groups = [
            {'params': list(model.token_embedding_table.parameters()) + list(model.position_embedding_table.parameters()), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) + 1))},
            {'params': model.regression_head.parameters(), 'lr': learning_rate}
        ] + [{'params': block.parameters(), 'lr': learning_rate * (LR_DECAY ** (len(model.blocks) - i))} for i, block in enumerate(model.blocks)]
        optimizer = torch.optim.AdamW(param_groups)

        # Initialize SI if updating
        si = None
        if args.update:
            # Load pre-trained model
            model = load_model_weights(model, 'model/model_weights.pth', device)

            # Initialize SI
            si = initialize_si(model, 'model/si_state.pth', lambda_si=lambda_si)

        # Load data and create DataLoader
        df = get_data()
        df = df[df['weighted_avg_720_hrs'] > 0]
        train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
        train_dataloader = prepare_dataloader(train_df, tokenizer)

        # Train the model
        print("Training Started!")
        train_model(model, optimizer, EPOCHS, device, train_dataloader, si=si)

        # Save the model
        os.makedirs('model', exist_ok=True)
        torch.save(model.state_dict(), 'model/model_weights.pth')
        print("Model weights saved.")

        # Save SI state
        if si is not None:
            si.save_state('model/si_state.pth')
            print("SI state saved.")

    elif args.mode == 'run':
        # Load the model weights
        model = load_model_weights(model, 'model/model_weights.pth', device)

        if args.test:
            # Load test data and run evaluation
            predictions, actuals = [], []
            for batch in test_dataloader:
                with torch.no_grad(), autocast():
                    outputs, _ = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch['labels'].cpu().numpy())

            # Calculate metrics
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Test MAE: {mae:.4f}, R2 Score: {r2:.4f}")

        else:
            if not args.input_text:
                raise ValueError("You must provide input text when mode is 'run' without --test")

            encoding = tokenizer(args.input_text, truncation=True, padding='max_length', max_length=block_size, return_tensors='pt').to(device)
            input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

            # Inference
            with torch.no_grad(), autocast():
                prediction, _ = model(input_ids, attention_mask)
            print(f"Predicted Price: {prediction.item()}")

if __name__ == "__main__":
    main()
