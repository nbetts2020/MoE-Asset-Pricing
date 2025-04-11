import os
import gc
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import logging
import subprocess
import re

from utils.utils import compute_l2_loss, upload_checkpoint_to_s3
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from deepspeed.runtime.zero.stage3 import GatheredParameters

# Import the helper function we defined in model.py to update RoPE buffers.
from model import update_model_rope_for_extended_context

# (Keep your extract_label_value function as is)
def extract_label_value(decoded_text):
    # This regex tries to find a number after the literal "[30 DAY LABEL]:"
    match = re.search(r'\<30 DAY LABEL\>:\s*([\d\.]+)', decoded_text)
    if match:
        return float(match.group(1))
    else:
        return None

def train_model(
    model,
    optimizer,  # Single optimizer object (or a dict of optimizers)
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
    Training loop for next-token prediction pretraining only.
    This function first performs normal pretraining with the standard context length,
    saves the checkpoint, then updates config.BLOCK_SIZE to 65536 and rebuilds the
    RoPE buffers, proceeding to run continual pretraining with the extended (64k) context.
    
    DeepSpeed or standard PyTorch training is supported.
    """
    if use_deepspeed:
        engine = model  # DeepSpeed engine passed from main.py
    else:
        adam_optimizer = optimizer

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    total_steps = len(dataloader) * epochs
    logging.info(f"Starting normal pretraining on rank {rank}.")
    logging.info(f"Dataloader length: {len(dataloader)} batches.")
    logging.info(f"Token embedding shape: {model.token_embedding_table.weight.shape}")

    ###########################################################################
    # Normal Next-token Prediction Pretraining
    ###########################################################################
    PRETRAIN_EPOCHS = epochs  # Use all epochs for next-token prediction
    logging.info("=== Normal Next-token Prediction Pretraining ===")
    pretrain_total_steps = len(dataloader) * PRETRAIN_EPOCHS

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        logging.info(f"--- Epoch {epoch}/{PRETRAIN_EPOCHS} ---")
        total_batches = len(dataloader)
        epoch_loss = 0.0
        running_avg_loss = 0.0
        alpha = 0.9

        for step, batch in enumerate(dataloader):
            current_step = (epoch - 1) * total_batches + step
            percent_complete = current_step / pretrain_total_steps

            print(f"[Pretrain] Processing batch {step + 1}/{total_batches}")
            input_ids = batch['input_ids'].to(device)

            # Next-token prediction branch with explicit BF16 gathering.
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
                logging.info(f"[Pretrain] Epoch {epoch}, Batch {step + 1}/{total_batches}, "
                             f"Loss: {batch_loss:.4f}, Running Avg Loss: {running_avg_loss:.4f}")
                # Debug Block: Print filtered tokens and compute MSE for the label.
                model.eval()
                with torch.no_grad():
                    B, T = input_ids.shape
                    if T > model.block_size:
                        input_ids_trim = input_ids[:, -model.block_size:]
                        T_trim = model.block_size
                    else:
                        input_ids_trim = input_ids
                        T_trim = T
                    pad_id = int(model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
                    attn_mask = (input_ids_trim != pad_id)
                    tok_emb = model.token_embedding_table(input_ids_trim)
                    pos_emb = model.position_embedding_table(torch.arange(T_trim, device=device))
                    x = tok_emb + pos_emb
                    for block in model.blocks:
                        x, _ = block(x, attn_mask)
                    x = model.ln_f(x)
                    shift_embeddings = x[:, :-1, :].reshape(-1, x.size(-1))
                    with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                        classifier = model.token_embedding_table.weight.clone()
                    logits = shift_embeddings @ classifier.T
                    predicted_ids = logits.argmax(dim=1).reshape(input_ids_trim.size(0), -1)
                    for i in range(min(2, input_ids_trim.size(0))):
                        active_input_tokens = [token for token, m in zip(input_ids_trim[i].tolist(), attn_mask[i].tolist()) if m]
                        active_target_tokens = [token for token, m in zip(input_ids_trim[i, 1:].tolist(), attn_mask[i, 1:].tolist()) if m]
                        active_pred_tokens = [token for token, m in zip(predicted_ids[i].tolist(), attn_mask[i, 1:].tolist()) if m]
                        input_text = model.tokenizer.decode(active_input_tokens)
                        target_text = model.tokenizer.decode(active_target_tokens)
                        pred_text = model.tokenizer.decode(active_pred_tokens)
                        print("Filtered Input tokens:    ", input_text)
                        print("Filtered Target tokens:   ", target_text)
                        print("Filtered Predicted tokens:", pred_text)
                        true_label = extract_label_value(target_text)
                        pred_label = extract_label_value(pred_text)
                        if true_label is not None and pred_label is not None:
                            mse = (pred_label - true_label) ** 2
                            print(f"MSE for sample {i}: {mse}")
                        else:
                            print(f"Could not extract label value for sample {i}")
                model.train()

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
        logging.info(f"[Pretrain] Finished Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

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
        raise RuntimeError(f"Rank {rank} failed to save pretrained checkpoint: {str(e)}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if use_deepspeed and rank == 0:
        consolidated_path = os.path.join(args.save_dir, "consolidated_" + pretrain_tag + ".pth")
        state_dict = model.module.state_dict()
        torch.save(state_dict, consolidated_path)
        if os.path.exists(consolidated_path):
            logging.info(f"Pretrained consolidated weights saved to {consolidated_path}")
        else:
            logging.error(f"Failed to save pretrained consolidated weights at {consolidated_path}")
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
    logging.info("Normal pretraining phase complete.")

    ###########################################################################
    # Continual Pretraining with Extended Context (64k tokens)
    ###########################################################################
    logging.info("=== Starting Continual Pretraining Phase with Extended Context ===")
    # Update the configuration: here we update BLOCK_SIZE for extended context.
    config.BLOCK_SIZE = 65536

    # Update the tokenizer's max length if needed:
    model.tokenizer.model_max_length = config.CONTEXT_WINDOW = 65536

    # Rebuild RoPE buffers in each transformer block to handle extended context.
    update_model_rope_for_extended_context(model, new_seq_len=65536, base=500000.0)

    # (Assume you create a new dataloader that produces 64k token sequences, e.g. via prepare_ft_dataloader)
    continual_dataloader = args.continual_dataloader  # This should be prepared externally

    CONTINUAL_EPOCHS = args.continual_epochs if hasattr(args, "continual_epochs") else 1
    continual_total_steps = len(continual_dataloader) * CONTINUAL_EPOCHS
    logging.info(f"Continual Pretraining will run for {CONTINUAL_EPOCHS} epochs, {len(continual_dataloader)} batches per epoch.")

    for epoch in range(1, CONTINUAL_EPOCHS + 1):
        model.train()
        logging.info(f"--- Continual Epoch {epoch}/{CONTINUAL_EPOCHS} ---")
        total_batches = len(continual_dataloader)
        epoch_loss = 0.0
        running_avg_loss = 0.0

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
