# utils/test.py

import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.utils import get_data, prepare_dataloader
from sklearn.model_selection import train_test_split
import random
import json
import logging
from utils.train import train_model
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.config import *

import numpy as np

def test_forgetting(model, optimizer, epochs, device, tokenizer, args, si=None, replay_buffer=None):
    """
    Test the model for catastrophic forgetting across multiple tasks.

    Args:
        model (nn.Module): The model to test.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of epochs for training each task.
        device (torch.device): Device (CPU or GPU).
        tokenizer: Tokenizer for data preparation.
        args: Argument parser object containing command-line arguments.
        si (SynapticIntelligence): SI object for regularization (optional).
        replay_buffer (MemoryReplayBuffer): Replay buffer for experience replay (optional).

    Returns:
        results (dict): Performance metrics for each task across all stages.
    """
    results = {}
    model.train()

    # Random seed for reproducibility
    random_seed = args.random_seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Load data
    df = get_data(percent_data=args.percent_data)
    df = df[df['weighted_avg_720_hrs'] > 0]  # Ensure valid market data

    # Select sectors with more than 1000 samples
    sector_counts = df['Sector'].value_counts()
    eligible_sectors = sector_counts[sector_counts >= 1000].index.tolist()

    if len(eligible_sectors) < args.num_tasks:
        raise ValueError(f"Not enough sectors with at least 1000 samples. Found {len(eligible_sectors)}, but num_tasks={args.num_tasks}")

    # Randomly select k sectors
    selected_sectors = random.sample(eligible_sectors, args.num_tasks)
    print(f"Selected sectors: {selected_sectors}")
    logging.info(f"Selected sectors: {selected_sectors}")

    # sectors_file = os.path.join(args.save_dir, 'selected_sectors.json')
    # with open(sectors_file, 'w') as f:
    #     json.dump(selected_sectors, f)
    # logging.info(f"Selected sectors saved to {sectors_file}")

    tasks = []

    # Prepare data loaders for each task
    for sector in selected_sectors:
        sector_df = df[df['Sector'] == sector]
        train_df, test_df = train_test_split(sector_df, test_size=0.15, random_state=random_seed)
        train_dataloader = prepare_dataloader(train_df, tokenizer, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = prepare_dataloader(test_df, tokenizer, batch_size=BATCH_SIZE, shuffle=False)
        tasks.append({
            'sector': sector,
            'train_dataloader': train_dataloader,
            'test_dataloader': test_dataloader
        })

    # Initialize SI if provided
    if si:
        si.initialize(model)  # Initialize SI once before training

    # Initialize replay buffer if provided
    if replay_buffer:
        replay_buffer.initialize()

    # Dictionary to store initial metrics for percentage change calculation
    initial_metrics = {}

    # Sequential Training and Evaluation
    for i, task in enumerate(tasks):
        sector = task['sector']
        print(f"\nTraining on Task {i + 1}: Sector '{sector}'")
        logging.info(f"Training on Task {i + 1}: Sector '{sector}'")

        # Train on current task
        train_model(
            model=model,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            dataloader=task['train_dataloader'],
            si=si,
            accumulation_steps=1,  # Adjust if needed
            replay_buffer=replay_buffer,
            test_dataloader=None  # Optionally provide test_dataloader
        )

        # Evaluate on all tasks seen so far
        for j, prev_task in enumerate(tasks[:i + 1]):
            prev_sector = prev_task['sector']
            print(f"Evaluating on Task {j + 1}: Sector '{prev_sector}' after training on Task {i + 1}")
            logging.info(f"Evaluating on Task {j + 1}: Sector '{prev_sector}' after training on Task {i + 1}")
            mse, rmse, mae, r2 = evaluate_task(model, prev_task['test_dataloader'], device)
            print(f"Task {j + 1} - Sector '{prev_sector}' - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            logging.info(f"Task {j + 1} - Sector '{prev_sector}' - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

            # Calculate percentage changes
            task_key = f"Task_{j + 1}_Sector_{prev_sector}"
            if task_key not in initial_metrics:
                # Store initial metrics
                initial_metrics[task_key] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

            # Percentage change from initial metrics
            initial_mse = initial_metrics[task_key]['mse']
            initial_rmse = initial_metrics[task_key]['rmse']
            initial_mae = initial_metrics[task_key]['mae']
            initial_r2 = initial_metrics[task_key]['r2']

            mse_pct_change = ((mse - initial_mse) / initial_mse) * 100 if initial_mse != 0 else 0
            rmse_pct_change = ((rmse - initial_rmse) / initial_rmse) * 100 if initial_rmse != 0 else 0
            mae_pct_change = ((mae - initial_mae) / initial_mae) * 100 if initial_mae != 0 else 0
            r2_pct_change = ((r2 - initial_r2) / initial_r2) * 100 if initial_r2 != 0 else 0

            # Store results
            if task_key not in results:
                results[task_key] = []
            results[task_key].append({
                'trained_on_task': i + 1,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mse_pct_change_from_initial': mse_pct_change,
                'rmse_pct_change_from_initial': rmse_pct_change,
                'mae_pct_change_from_initial': mae_pct_change,
                'r2_pct_change_from_initial': r2_pct_change
            })

            # Optional: Print percentage change from last iteration
            if len(results[task_key]) > 1:
                prev_metrics = results[task_key][-2]
                prev_mse = prev_metrics['mse']
                prev_rmse = prev_metrics['rmse']
                prev_mae = prev_metrics['mae']
                prev_r2 = prev_metrics['r2']

                mse_pct_change_last = ((mse - prev_mse) / prev_mse) * 100 if prev_mse != 0 else 0
                rmse_pct_change_last = ((rmse - prev_rmse) / prev_rmse) * 100 if prev_rmse != 0 else 0
                mae_pct_change_last = ((mae - prev_mae) / prev_mae) * 100 if prev_mae != 0 else 0
                r2_pct_change_last = ((r2 - prev_r2) / prev_r2) * 100 if prev_r2 != 0 else 0

                # Add percentage changes from last iteration to the current result
                results[task_key][-1]['mse_pct_change_from_last'] = mse_pct_change_last
                results[task_key][-1]['rmse_pct_change_from_last'] = rmse_pct_change_last
                results[task_key][-1]['mae_pct_change_from_last'] = mae_pct_change_last
                results[task_key][-1]['r2_pct_change_from_last'] = r2_pct_change_last

                print(f"Change in MSE for Task {j + 1} (Sector '{prev_sector}') after training on Task {i + 1}: {mse_pct_change_last:.2f}%")
                logging.info(f"Change in MSE for Task {j + 1} (Sector '{prev_sector}') after training on Task {i + 1}: {mse_pct_change_last:.2f}%")
            else:
                # For the first result, percentage change from last is zero
                results[task_key][-1]['mse_pct_change_from_last'] = 0
                results[task_key][-1]['rmse_pct_change_from_last'] = 0
                results[task_key][-1]['mae_pct_change_from_last'] = 0
                results[task_key][-1]['r2_pct_change_from_last'] = 0

    return results

def evaluate_task(model, dataloader, device):
    """
    Evaluate the model on a given task (dataloader).

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Dataloader for the task to evaluate.
        device (torch.device): Device (CPU or GPU).

    Returns:
        mse (float): Mean Squared Error.
        rmse (float): Root Mean Squared Error.
        mae (float): Mean Absolute Error.
        r2 (float): R-squared score.
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs, _ = model(input_ids=input_ids)
            else:
                outputs, _ = model(input_ids=input_ids)

            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # Ensure outputs_np and labels_np are 1D arrays
            outputs_np = outputs_np.reshape(-1)
            labels_np = labels_np.reshape(-1)

            predictions.extend(outputs_np.tolist())
            actuals.extend(labels_np.tolist())

    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    model.train()  # Set the model back to train mode
    return mse, rmse, mae, r2
