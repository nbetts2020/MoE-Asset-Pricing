import os
import gc
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import logging
import subprocess

from utils.utils import compute_l2_loss, upload_checkpoint_to_s3, save_ebm_model
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer

def train_model(
    model,
    optimizers,
    epochs,
    device,
    dataloader,
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
    Single-dataloader training loop for SC454k-formatted dataset.
    Integrates:
      - L2 regularization (args.use_l2, args.lambda_l2)
      - Online Learning with SI (si)
      - Elastic Weight Consolidation (ewc)
      - Memory Replay Buffer (replay_buffer)
      - Optional EBM placeholders
      - DeepSpeed or standard PyTorch training
    Handles grad=None for sparse MoE models with DeepSpeed.
    """
    if use_deepspeed:
        engine = model  # DeepSpeed engine passed from main.py
        engine_optimizer = optimizers[0]  # Single optimizer from DeepSpeed
    else:
        adam_optimizer, muon_optimizer = optimizers

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    logging.info(f"Beginning training for {epochs} epoch(s) on rank {rank}.")
    logging.info(f"Dataloader length: {len(dataloader)} batches.")
    logging.info(f"Token embedding shape: {model.token_embedding_table.weight.shape}")

    for epoch in range(1, epochs + 1):
        model.train()
        logging.info(f"=== Starting epoch {epoch}/{epochs} ===")
        total_batches = len(dataloader)

        for step, batch in enumerate(dataloader):
            print(f"Processing batch {step + 1}/{total_batches}")

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=True):  # Updated for PyTorch 2.x
                outputs, _ = model(input_ids=input_ids)
                loss = F.mse_loss(outputs.squeeze(-1), labels.float())

                if args.use_l2:
                    loss += args.lambda_l2 * compute_l2_loss(model)

                if replay_buffer and len(replay_buffer.buffer) > 0:
                    replay_loss = replay_buffer.replay_and_calculate_loss(
                        model=model,
                        tokenizer=tokenizer,
                        replay_batch_size=args.replay_batch_size,
                        device=device,
                        alpha=args.replay_buffer_weight
                    )
                    loss += replay_loss

                if args.use_ewc and ewc:
                    for ewc_instance in ewc:
                        loss += args.lambda_ewc * ewc_instance.penalty(model)

                if args.use_si and si:
                    loss += si.penalty(model)

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                for name, param in engine.module.named_parameters():
                    if param.requires_grad and param.grad is None:
                        param.grad = torch.zeros_like(param, device=device)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                muon_optimizer.zero_grad()
                loss.backward()
                adam_optimizer.step()
                muon_optimizer.step()

            if args.use_si and si:
                si.update_weights(model)

            if args.use_replay_buffer and replay_buffer:
                replay_buffer.add_batch(batch)

        logging.info(f"=== Finished epoch {epoch} ===")

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
        model.save_checkpoint(args.save_dir, tag=tag, client_state={})
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to save checkpoint: {str(e)}")

    # Ensure all ranks wait until saving is complete
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Rank 0 handles S3 upload and EBM saving
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
        if args.use_ebm and ebm is not None:
            save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models", args=args)

    logging.info("All epochs completed.")
