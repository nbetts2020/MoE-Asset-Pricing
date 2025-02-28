# train.py

import os
import gc
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import logging
from tqdm import tqdm

from utils.utils import (
    get_data,
    prepare_dataloader,
    compute_l2_loss
)
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer

def train_model(
    model,
    optimizers,
    epochs,
    device,
    dataloader,   # Not used in chunk mode, kept for API compatibility
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    ebm=None,
    ebm_optimizer=None,
    tokenizer=None,
    use_deepspeed=False
):
    """
    Example chunk-based training using global_offset/global_max logic to limit usage
    to (args.percent_data)% of the *training portion* of the dataset.
    Includes tqdm progress bars for monitoring progress.
    """

    adam_optimizer, muon_optimizer = optimizers
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    # Calculate global_max based on your stated logic:
    #   total_rows = 453932 (full dataset)
    #   training split = 80% => 0.8 * 453932
    #   then we only want (args.percent_data / 100) of that training split
    total_for_training = 0.8 * 453932
    global_max = int((args.percent_data / 100.0) * total_for_training)

    # We'll track how many rows we've used so far
    global_offset = 0

    logging.info(f"Using up to {global_max} rows from the training set, "
                 f"which is {args.percent_data}% of the 80% split (total 453,932).")

    for epoch in range(1, epochs + 1):
        model.train()
        logging.info(f"=== Starting epoch {epoch}/{epochs} ===")
        
        # Use tqdm to iterate over the 13 expected chunks
        for window_index in tqdm(range(1, 14), desc=f"Epoch {epoch} chunks"):
            # Build the DataLoader for this chunk
            loader = prepare_dataloader(
                epoch=epoch,
                window_index=window_index,
                tokenizer=tokenizer,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                global_offset=global_offset,
                global_max=global_max,
                args=args
            )

            # If empty, it means either chunk is empty or we've hit global_max
            if len(loader) == 0:
                logging.info(
                    f"No data returned for window_index={window_index} "
                    f"(epoch {epoch}). Possibly reached the global_max. Stopping chunk loop."
                )
                break

            # Wrap the batch loop in tqdm for progress on current chunk
            for step, batch in enumerate(tqdm(loader, desc=f"Chunk {window_index} batches", leave=False)):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                with torch.cuda.amp.autocast(enabled=True):
                    outputs, _ = model(input_ids=input_ids)
                    loss = F.mse_loss(outputs.squeeze(-1), labels.float())

                    # Optional: L2 Regularization
                    if args.use_l2:
                        loss += args.lambda_l2 * compute_l2_loss(model)

                    # Optional: Replay Buffer loss
                    if replay_buffer and len(replay_buffer) > 0:
                        replay_loss = replay_buffer.replay_and_calculate_loss(
                            model,
                            tokenizer,
                            args.replay_batch_size,
                            device,
                            alpha=args.replay_buffer_weight
                        )
                        loss += replay_loss

                    # Optional: EWC Regularization
                    if args.use_ewc and ewc:
                        for ewc_instance in ewc:
                            loss += args.lambda_ewc * ewc_instance.penalty(model)

                    # Optional: SI Regularization
                    if args.use_si and si:
                        loss += si.penalty(model)

                adam_optimizer.zero_grad()
                muon_optimizer.zero_grad()

                if use_deepspeed and hasattr(model, "backward"):
                    model.backward(loss)
                    model.step()
                else:
                    loss.backward()
                    adam_optimizer.step()
                    muon_optimizer.step()

                # SI online update
                if args.use_si and si:
                    si.update_weights(model)

                # Add batch to replay buffer
                if args.use_replay_buffer and replay_buffer:
                    replay_buffer.add_batch(batch)

            # Update global_offset by however many rows were in this chunk
            if hasattr(loader.dataset, 'df'):
                num_rows = len(loader.dataset.df)
                global_offset += num_rows
                logging.info(f"Updated global_offset to {global_offset} (added {num_rows} rows)")

            # Cleanup for this chunk
            loader = None
            gc.collect()

            # If we've now used up the entire global_max, break from chunk loop
            if global_offset >= global_max:
                logging.info("Reached global_max rows. Stopping further chunk loading.")
                break

        # End of epoch
        logging.info(f"=== Finished epoch {epoch} ===")

        # EWC consolidate
        if args.use_ewc and ewc:
            for ewc_instance in ewc:
                ewc_instance.consolidate(model)

        # SI finalize
        if args.use_si and si:
            si.update_omega(model)

        # If we've used all rows, no need to start a new epoch
        if global_offset >= global_max:
            logging.info("global_offset already >= global_max, stopping training entirely.")
            break

    logging.info("All epochs completed (or hit global data limit).")
