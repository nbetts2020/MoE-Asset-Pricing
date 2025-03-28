import os
import gc
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import logging
import subprocess

from utils.utils import compute_l2_loss, upload_checkpoint_to_s3
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer

def train_model(
    model,
    optimizer,  # Single optimizer object
    epochs,
    device,
    dataloader,
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    tokenizer=None,
    use_deepspeed=False
):
    """
    Single-dataloader training loop for SC454k-formatted dataset.
    Integrates:
      - L2 regularization (args.use_l2, args.lambda_l2)
      - Online Learning with SI (si)
      - Elastic Weight Consolidation (ewc)
      - Memory Replay Buffer (replay_buffer)
      - EBM now integrated into model.forward()
      - DeepSpeed or standard PyTorch training
    Handles grad=None for sparse MoE models with DeepSpeed.

    Args:
        model: The model to train (DeepSpeed engine if use_deepspeed=True, else PyTorch model)
        optimizer: Single optimizer instance (used directly in non-DeepSpeed case)
        epochs: Number of training epochs
        device: Device to run training on (e.g., 'cuda' or 'cpu')
        dataloader: DataLoader providing training batches
        args: Command-line arguments from argparse
        si: SynapticIntelligence instance (optional)
        ewc: List of ElasticWeightConsolidation instances (optional)
        replay_buffer: MemoryReplayBuffer instance (optional)
        tokenizer: Tokenizer for replay buffer (optional)
        use_deepspeed: Boolean flag to enable DeepSpeed training
    """
    if use_deepspeed:
        engine = model  # DeepSpeed engine passed from main.py
    else:
        adam_optimizer = optimizer  # Single optimizer for non-DeepSpeed case

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    logging.info(f"Beginning training for {epochs} epoch(s) on rank {rank}.")
    logging.info(f"Dataloader length: {len(dataloader)} batches.")
    logging.info(f"Token embedding shape: {model.token_embedding_table.weight.shape}")

    total_steps = len(dataloader) * epochs
    for epoch in range(1, epochs + 1):
        model.train()
        logging.info(f"=== Starting epoch {epoch}/{epochs} ===")
        total_batches = len(dataloader)
        epoch_loss = 0.0  # Track total loss for the epoch
        running_avg_loss = 0.0  # Initialize running average
        alpha = 0.9  # Smoothing factor for EMA (0.9 = 10-batch smoothing)

        for step, batch in enumerate(dataloader):
            current_step = (epoch - 1) * total_batches + step
            percent_complete = current_step / total_steps

            print(f"Processing batch {step + 1}/{total_batches}")

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=True):  # Updated for PyTorch 2.x
                # Model now returns output, ebm_energy, and loss
                output, ebm_energy, loss = model(
                    input_ids=input_ids,
                    targets=labels,
                    percent_complete=percent_complete,
                    use_entropy_reg=args.use_entropy_reg  # Pass if needed
                )

                # Add L2 regularization if enabled
                if args.use_l2:
                    loss += args.lambda_l2 * compute_l2_loss(model)

                # Add replay buffer loss if enabled
                if replay_buffer and len(replay_buffer.buffer) > 0:
                    replay_loss = replay_buffer.replay_and_calculate_loss(
                        model=model,
                        tokenizer=tokenizer,
                        replay_batch_size=args.replay_batch_size,
                        device=device,
                        alpha=args.replay_buffer_weight
                    )
                    loss += replay_loss

                # Add EWC penalty if enabled
                if args.use_ewc and ewc:
                    for ewc_instance in ewc:
                        loss += args.lambda_ewc * ewc_instance.penalty(model)

                # Add SI penalty if enabled
                if args.use_si and si:
                    loss += si.penalty(model)

            # Log loss for this batch
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Update running average
            if step == 0 and epoch == 1:
                running_avg_loss = batch_loss  # First batch initializes directly
            else:
                running_avg_loss = alpha * running_avg_loss + (1 - alpha) * batch_loss
            if step % 10 == 0:
                logging.info(f"Outputs: {output[:5]}, Labels: {labels[:5]}, EBM Energy: {ebm_energy[:5]}")
            # Log batch loss and running average
            logging.info(f"Epoch {epoch}, Batch {step + 1}/{total_batches}, "
                        f"Loss: {batch_loss:.4f}, Running Avg Loss: {running_avg_loss:.4f}")

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                for name, param in engine.module.named_parameters():
                    if param.requires_grad and param.grad is None:
                        param.grad = torch.zeros_like(param, device=device)
                engine.step()  # This internally calls the optimizer's step
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()

            if args.use_si and si:
                si.update_weights(model)

            if args.use_replay_buffer and replay_buffer:
                replay_buffer.add_batch(batch)

        avg_epoch_loss = epoch_loss / total_batches
        logging.info(f"=== Finished epoch {epoch}, Average Loss: {avg_epoch_loss:.4f} ===")

        if args.use_ewc and ewc:
            for ewc_instance in ewc:
                ewc_instance.consolidate(model)

        if args.use_si and si:
            si.update_omega(model)

        gc.collect()

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Save Checkpoint with explicit rank-specific handling
    tag = "final"
    checkpoint_dir = os.path.join(args.save_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        if use_deepspeed:
            model.save_checkpoint(args.save_dir, tag=tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to save checkpoint: {str(e)}")

    # Ensure all ranks wait until saving is complete
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Save consolidated model weights on rank 0 (DeepSpeed case)
    if use_deepspeed and rank == 0:
        consolidated_path = os.path.join(args.save_dir, "consolidated_final.pth")
        state_dict = model.module.state_dict()  # Extract model weights directly
        torch.save(state_dict, consolidated_path)
        if os.path.exists(consolidated_path):
            logging.info(f"Consolidated model weights saved to {consolidated_path}")
        else:
            logging.error(f"Failed to save consolidated model weights at {consolidated_path}")

    # Rank 0 handles S3 upload
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")

    logging.info("All epochs completed.")
