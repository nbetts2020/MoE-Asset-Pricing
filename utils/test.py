# utils/test.py

import torch
from sklearn.metrics import mean_squared_error, r2_score
from utils.utils import get_data, prepare_dataloader
from sklearn.model_selection import train_test_split
import random
import json  # for saving results

def test_forgetting(model, optimizer, epochs, device, tokenizer, args, si=None):
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

    Returns:
        results (dict): Performance metrics for each task across all stages.
    """
    results = {}
    model.train()

    # Set the random seed for reproducibility
    random_seed = args.random_seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Load data
    df = get_data()
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

    # Optional: Save the selected sectors to a file
    sectors_file = os.path.join(args.save_dir, 'selected_sectors.json')
    with open(sectors_file, 'w') as f:
        json.dump(selected_sectors, f)
    logging.info(f"Selected sectors saved to {sectors_file}")

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
        si.initialize(model)

    # Sequential Training and Evaluation
    for i, task in enumerate(tasks):
        sector = task['sector']
        print(f"\nTraining on Task {i + 1}: Sector '{sector}'")
        logging.info(f"Training on Task {i + 1}: Sector '{sector}'")
        train_model(model, optimizer, epochs, device, task['train_dataloader'], si=si)

        # Evaluate on all tasks seen so far
        for j, prev_task in enumerate(tasks[:i + 1]):
            prev_sector = prev_task['sector']
            print(f"Evaluating on Task {j + 1}: Sector '{prev_sector}' after training on Task {i + 1}")
            logging.info(f"Evaluating on Task {j + 1}: Sector '{prev_sector}' after training on Task {i + 1}")
            predictions, actuals = evaluate_task(model, prev_task['test_dataloader'], device)
            mse = mean_squared_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            print(f"Task {j + 1} - Sector '{prev_sector}' - MSE: {mse:.4f}, R2: {r2:.4f}")
            logging.info(f"Task {j + 1} - Sector '{prev_sector}' - MSE: {mse:.4f}, R2: {r2:.4f}")

            # Store results
            task_key = f"Task_{j + 1}_Sector_{prev_sector}"
            if task_key not in results:
                results[task_key] = []
            results[task_key].append({
                'trained_on_task': i + 1,
                'mse': mse,
                'r2': r2
            })

            # Optional: Print change in performance if not the first evaluation
            if len(results[task_key]) > 1:
                prev_mse = results[task_key][-2]['mse']
                mse_change = mse - prev_mse
                print(f"Change in MSE for Task {j + 1} (Sector '{prev_sector}') after training on Task {i + 1}: {mse_change:.4f}")
                logging.info(f"Change in MSE for Task {j + 1} (Sector '{prev_sector}') after training on Task {i + 1}: {mse_change:.4f}")

    return results

def evaluate_task(model, dataloader, device):
    """
    Evaluate the model on a given task (dataloader).

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Dataloader for the task to evaluate.
        device (torch.device): Device (CPU or GPU).

    Returns:
        predictions (list): Model predictions.
        actuals (list): Ground truth labels.
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(input_ids=input_ids, targets=labels)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    return predictions, actuals
