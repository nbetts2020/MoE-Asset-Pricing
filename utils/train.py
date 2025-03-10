import os
import gc
import time  # for sleep
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

    Final checkpoint saving, S3 upload, and EBM saving occur at the end.
    """
    adam_optimizer, muon_optimizer = optimizers
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    logging.info(f"Beginning training for {epochs} epoch(s).")
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

            with torch.cuda.amp.autocast(enabled=True):
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

            adam_optimizer.zero_grad()
            muon_optimizer.zero_grad()

            if use_deepspeed and hasattr(model, "backward"):
                model.backward(loss)
                model.step()
            else:
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

    tag = "final"
    model.save_checkpoint(args.save_dir, tag=tag)  # All ranks save their shards
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # Wait for all ranks to finish saving
    if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        # In train_model, just before saving the checkpoint
        logging.info(f"Token embedding shape: {model.token_embedding_table.weight.shape}")
        logging.info(f"Embedding dimension (n_embed): {model.token_embedding_table.weight.size(1)}")
        logging.info(f"DeepSpeed ZeRO checkpoint saved to {args.save_dir}, tag={tag}")
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
        if args.use_ebm and ebm is not None:
            save_ebm_model(ebm, epoch=config.EPOCHS, save_dir="models", args=args)
            logging.info("EBM model saved.")

    logging.info("All epochs completed.")
