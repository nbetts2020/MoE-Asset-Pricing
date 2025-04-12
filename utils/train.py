import os
import gc
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import logging
import subprocess
import re

from utils.utils import compute_l2_loss, upload_checkpoint_to_s3, prepare_ft_dataloader
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from deepspeed.runtime.zero.stage3 import GatheredParameters

# Import the helper function to update RoPE buffers.
from utils.model import update_model_rope_for_extended_context

# (Keep your extract_label_value function as is)
def extract_label_value(decoded_text):
    """
    Extracts the numerical label following the "<30 DAY LABEL>:" marker and returns it as a float.
    If multiple dots are found (e.g. "2..98"), they are replaced with a single dot.
    Returns None if extraction or conversion fails.
    """
    match = re.search(r'\<30 DAY LABEL\>:\s*([\d\.]+)', decoded_text)
    if match:
        num_str = match.group(1)
        # Replace multiple dots with a single dot.
        num_str = re.sub(r'\.\.+', '.', num_str)
        try:
            return float(num_str)
        except ValueError as e:
            logging.error(f"Could not convert label string to float: '{num_str}'. Error: {e}")
            return None
    else:
        return None

def train_model(
    model,
    optimizer,      # Single optimizer object (or dict of optimizers)
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
    Training loop for next-token prediction pretraining, followed by two fine-tuning stages:
      1. Normal pretraining phase using ft_dataset_1 (stage_1=True).
      2. Continual pretraining (interpolation fine-tuning) phase with extended context.
      3. Supervised fine-tuning phase using the new dataset (ft_dataset_2, stage_1=False).

    DeepSpeed or standard PyTorch training is supported.
    """
    if use_deepspeed:
        engine = model  # DeepSpeed engine passed from main.py
    else:
        adam_optimizer = optimizer

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    ############################################################################
    # PHASE 1: Normal Pretraining (using ft_dataset_1)
    ############################################################################
    logging.info(f"Starting normal pretraining on rank {rank}.")
    logging.info(f"Dataloader length: {len(dataloader)} batches.")
    logging.info(f"Token embedding shape: {model.token_embedding_table.weight.shape}")

    PRETRAIN_EPOCHS = epochs  # Using 'epochs' for normal pretraining.
    pretrain_total_steps = len(dataloader) * PRETRAIN_EPOCHS

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        logging.info(f"--- Normal Pretraining Epoch {epoch}/{PRETRAIN_EPOCHS} ---")
        total_batches = len(dataloader)
        epoch_loss = 0.0
        running_avg_loss = 0.0
        alpha = 0.9

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            with torch.amp.autocast('cuda', enabled=False):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().to(torch.bfloat16)
                    loss = model.forward_next_token_efficient(input_ids, reduction="mean", force_bf16=True)

            batch_loss = loss.item()
            epoch_loss += batch_loss
            if step == 0 and epoch == 1:
                running_avg_loss = batch_loss
            else:
                running_avg_loss = alpha * running_avg_loss + (1 - alpha) * batch_loss

            if step % 10 == 0:
                logging.info(f"[Normal Pretrain] Epoch {epoch}, Batch {step + 1}/{total_batches}, "
                             f"Loss: {batch_loss:.4f}, Running Avg Loss: {running_avg_loss:.4f}")
                # (Optional debug block omitted here for brevity.)

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                for name, param in engine.module.named_parameters():
                    if param.requires_grad and param.grad is None:
                        param.grad = torch.zeros_like(param, device=device)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()

            # Optionally update online learning penalties:
            if args.use_si and si:
                si.update_weights(model)
            if args.use_ewc and ewc:
                for ewc_instance in ewc:
                    loss += args.lambda_ewc * ewc_instance.penalty(model)
            if replay_buffer and len(replay_buffer.buffer) > 0:
                replay_loss = replay_buffer.replay_and_calculate_loss(
                    model=model,
                    tokenizer=tokenizer,
                    replay_batch_size=args.replay_batch_size,
                    device=device,
                    alpha=args.replay_buffer_weight
                )
                loss += replay_loss

        avg_epoch_loss = epoch_loss / total_batches
        logging.info(f"[Normal Pretrain] Finished Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

    # Save checkpoint from normal pretraining
    pretrain_tag = "normal_pretrained"
    pretrain_dir = os.path.join(args.save_dir, pretrain_tag)
    os.makedirs(pretrain_dir, exist_ok=True)
    try:
        if use_deepspeed:
            model.save_checkpoint(args.save_dir, tag=pretrain_tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(pretrain_dir, "model.pth"))
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to save normal pretrained checkpoint: {str(e)}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if use_deepspeed and rank == 0:
        consolidated_path = os.path.join(args.save_dir, "consolidated_" + pretrain_tag + ".pth")
        state_dict = model.module.state_dict()
        torch.save(state_dict, consolidated_path)
        if os.path.exists(consolidated_path):
            logging.info(f"Normal pretrained consolidated weights saved to {consolidated_path}")
        else:
            logging.error(f"Failed to save normal pretrained consolidated weights at {consolidated_path}")
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
    logging.info("Normal pretraining phase complete.")

    ############################################################################
    # PHASE 2: Continual Pretraining (Interpolation Fine-tuning) with Extended Context
    ############################################################################
    logging.info("=== Starting Continual Pretraining Phase with Extended Context ===")
    # Update configuration: set BLOCK_SIZE (and CONTEXT_WINDOW) to 65536.
    config.BLOCK_SIZE = 65536
    model.tokenizer.model_max_length = config.CONTEXT_WINDOW = config.BLOCK_SIZE

    # Rebuild RoPE buffers in each transformer block to handle extended context.
    update_model_rope_for_extended_context(model, new_seq_len=config.BLOCK_SIZE, base=500000.0)

    # Create a new dataloader for extended sequences (still using ft_dataset_1).
    continual_dataloader = prepare_ft_dataloader(tokenizer, config.BLOCK_SIZE, shuffle=False, args=args, stage_1=True)

    CONTINUAL_EPOCHS = 1
    logging.info(f"Continual Pretraining will run for {CONTINUAL_EPOCHS} epochs, {len(continual_dataloader)} batches per epoch.")

    for epoch in range(1, CONTINUAL_EPOCHS + 1):
        model.train()
        logging.info(f"--- Continual Epoch {epoch}/{CONTINUAL_EPOCHS} ---")
        total_batches = len(continual_dataloader)
        epoch_loss = 0.0

        for step, batch in enumerate(continual_dataloader):
            input_ids = batch['input_ids'].to(device)
            with torch.amp.autocast('cuda', enabled=False):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().to(torch.bfloat16)
                    loss = model.forward_next_token_efficient(input_ids, reduction="mean", force_bf16=True)

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if step % 10 == 0:
                logging.info(f"[Continual] Epoch {epoch}, Batch {step + 1}/{total_batches}, Loss: {batch_loss:.4f}")

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                for name, param in engine.module.named_parameters():
                    if param.requires_grad and param.grad is None:
                        param.grad = torch.zeros_like(param, device=device)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()

        avg_epoch_loss = epoch_loss / total_batches
        logging.info(f"[Continual] Finished Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

    # Save checkpoint from continual pretraining
    continual_tag = "continual_pretrained_64k"
    continual_dir = os.path.join(args.save_dir, continual_tag)
    os.makedirs(continual_dir, exist_ok=True)
    try:
        if use_deepspeed:
            model.save_checkpoint(args.save_dir, tag=continual_tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(continual_dir, "model.pth"))
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to save continual pretrained checkpoint: {str(e)}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if use_deepspeed and rank == 0:
        consolidated_path = os.path.join(args.save_dir, "consolidated_" + continual_tag + ".pth")
        state_dict = model.module.state_dict()
        torch.save(state_dict, consolidated_path)
        if os.path.exists(consolidated_path):
            logging.info(f"Continual pretrained consolidated weights saved to {consolidated_path}")
        else:
            logging.error(f"Failed to save continual pretrained consolidated weights at {consolidated_path}")
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
    logging.info("Continual pretraining phase complete.")

    ############################################################################
    # PHASE 3: Supervised Fine-tuning Stage (using ft_dataset_2)
    ############################################################################
    logging.info("=== Starting Supervised Fine-tuning Stage ===")
    # For supervised fine-tuning, we load new data by setting stage_1=False.
    # Depending on your fine-tuning design, you might want to revert the context length,
    # but here we assume you fine-tune on the extended context as well.
    ft_dataloader = prepare_ft_dataloader(tokenizer, config.BLOCK_SIZE, shuffle=True, args=args, stage_1=False)

    SUPERVISED_EPOCHS = 1
    logging.info(f"Supervised Fine-tuning will run for {SUPERVISED_EPOCHS} epochs, {len(ft_dataloader)} batches per epoch.")
    config.LEARNING_RATE = 1e-5
    config.LR_DECAY = 0.95
    config.DROPOUT = 0.05
    
    for epoch in range(1, SUPERVISED_EPOCHS + 1):
        model.train()
        logging.info(f"--- Supervised Fine-tuning Epoch {epoch}/{SUPERVISED_EPOCHS} ---")
        total_batches = len(ft_dataloader)
        epoch_loss = 0.0

        for step, batch in enumerate(ft_dataloader):
            input_ids = batch['input_ids'].to(device)
            # For supervised fine-tuning, you might want to use a lower learning rate
            # or different training objectives if needed.
            with torch.amp.autocast('cuda', enabled=False):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().to(torch.bfloat16)
                    loss = model.forward_next_token_efficient(input_ids, reduction="mean", force_bf16=True)

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if step % 10 == 0:
                logging.info(f"[Supervised FT] Epoch {epoch}, Batch {step + 1}/{total_batches}, Loss: {batch_loss:.4f}")

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                for name, param in engine.module.named_parameters():
                    if param.requires_grad and param.grad is None:
                        param.grad = torch.zeros_like(param, device=device)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()

        avg_epoch_loss = epoch_loss / total_batches
        logging.info(f"[Supervised FT] Finished Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

    # Save checkpoint from supervised fine-tuning
    ft_tag = "supervised_finetuned"
    ft_dir = os.path.join(args.save_dir, ft_tag)
    os.makedirs(ft_dir, exist_ok=True)
    try:
        if use_deepspeed:
            model.save_checkpoint(args.save_dir, tag=ft_tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(ft_dir, "model.pth"))
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to save supervised finetuned checkpoint: {str(e)}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if use_deepspeed and rank == 0:
        consolidated_path = os.path.join(args.save_dir, "consolidated_" + ft_tag + ".pth")
        state_dict = model.module.state_dict()
        torch.save(state_dict, consolidated_path)
        if os.path.exists(consolidated_path):
            logging.info(f"Supervised finetuned consolidated weights saved to {consolidated_path}")
        else:
            logging.error(f"Failed to save supervised finetuned consolidated weights at {consolidated_path}")
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
    logging.info("Supervised fine-tuning phase complete.")
