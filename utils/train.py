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
from deepspeed.runtime.zero.stage3 import GatheredParameters

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
    Training loop that automatically runs two phases in sequence:
      Phase 1: Next-token prediction pretraining.
               Sequences are truncated to block_size and the model uses forward_next_token().
               A checkpoint ("pretrained") is saved immediately afterwards.
      Phase 2: Regression fine-tuning.
               If sequence length exceeds block_size, clustering compresses the representation.
               The model uses forward() and applies attention pooling to produce a scalar output.

    Online learning (SI, EWC), L2 regularization, and replay buffer losses are applied in the regression phase.
    DeepSpeed or standard PyTorch training is supported.
    """
    if use_deepspeed:
        engine = model  # DeepSpeed engine passed from main.py
    else:
        adam_optimizer = optimizer

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    total_steps = len(dataloader) * epochs
    logging.info(f"Starting full training on rank {rank}.")
    logging.info(f"Dataloader length: {len(dataloader)} batches.")
    logging.info(f"Token embedding shape: {model.token_embedding_table.weight.shape}")

    ###########################################################################
    # Phase 1: Next-token Prediction Pretraining
    ###########################################################################
    PRETRAIN_EPOCHS = 1  # Hardcoded pretraining phase length
    logging.info("=== Phase 1: Next-token Prediction Pretraining ===")
    pretrain_total_steps = len(dataloader) * PRETRAIN_EPOCHS
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        logging.info(f"--- Pretraining Epoch {epoch}/{PRETRAIN_EPOCHS} ---")
        total_batches = len(dataloader)
        epoch_loss = 0.0
        running_avg_loss = 0.0
        alpha = 0.9

        for step, batch in enumerate(dataloader):
            current_step = (epoch - 1) * total_batches + step
            percent_complete = current_step / pretrain_total_steps

            print(f"[Pretrain] Processing batch {step + 1}/{total_batches}")
            input_ids = batch['input_ids'].to(device)

            # In pretraining, use the next-token prediction branch.
            # Fix for CCE data type mismatch in DeepSpeed ZeRO-3
            # Explicitly handle parameter gathering and precision for CCE compatibility
            with torch.amp.autocast('cuda', enabled=False):
                # Gather distributed parameters
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    # Explicitly convert to BF16 which CCE's backward pass requires
                    model._gathered_weights = model.token_embedding_table.weight.clone().to(torch.bfloat16)

                    # Forward in BF16 mode - no autocast, explicit conversion
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
                # --- Debug Block: Print token-level info without EOS padding ---
                model.eval()
                with torch.no_grad():
                    B, T = input_ids.shape
                    # Use right slicing to get the relevant block.
                    if T > model.block_size:
                        input_ids_trim = input_ids[:, -model.block_size:]
                        T_trim = model.block_size
                    else:
                        input_ids_trim = input_ids
                        T_trim = T
                    # Compute an attention mask (1 for non-pad tokens, 0 for pad tokens).
                    pad_id = int(model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
                    attn_mask = (input_ids_trim != pad_id)
                    # Now forward through the embedding layers.
                    tok_emb = model.token_embedding_table(input_ids_trim)
                    pos_emb = model.position_embedding_table(torch.arange(T_trim, device=device))
                    x = tok_emb + pos_emb
                    for block in model.blocks:
                        x, _ = block(x, attn_mask)
                    x = model.ln_f(x)
                    # Shift: tokens 0..T_trim-1 used to predict tokens 1..T_trim.
                    shift_embeddings = x[:, :-1, :].reshape(-1, x.size(-1))
                    with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                        classifier = model.token_embedding_table.weight.clone()
                    logits = shift_embeddings @ classifier.T
                    predicted_ids = logits.argmax(dim=1).reshape(input_ids_trim.size(0), -1)
                    # Filter out padded tokens using the attention mask.
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

        avg_epoch_loss = epoch_loss / total_batches
        logging.info(f"[Pretrain] Finished Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")

    # Save pretrained checkpoint after Phase 1
    pretrain_tag = "pretrained"
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
    logging.info("Pretraining phase complete.")

    ###########################################################################
    # Phase 2: Regression Fine-Tuning
    ###########################################################################
    finetune_epochs = epochs - PRETRAIN_EPOCHS if epochs > PRETRAIN_EPOCHS else 1
    logging.info("=== Phase 2: Regression Fine-Tuning ===")
    finetune_total_steps = len(dataloader) * finetune_epochs

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        logging.info(f"--- Finetuning Epoch {epoch}/{finetune_epochs} ---")
        total_batches = len(dataloader)
        epoch_loss = 0.0
        running_avg_loss = 0.0
        alpha = 0.9

        for step, batch in enumerate(dataloader):
            current_step = (epoch - 1) * total_batches + step
            percent_complete = current_step / finetune_total_steps

            print(f"[Finetune] Processing batch {step + 1}/{total_batches}")

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(
                    input_ids=input_ids,
                    targets=labels,
                    percent_complete=percent_complete,
                    use_entropy_reg=args.use_entropy_reg,
                    lambda_entropy=args.lambda_entropy
                )
                output = outputs["output"]
                ebm_energy = outputs["ebm_energy"]
                task_loss = outputs["task_loss"]
                ebm_loss = outputs["ebm_loss"]
                entropy_loss = outputs["entropy_loss"]

                loss = task_loss + (config.LAMBDA_EBM * percent_complete * ebm_loss)
                if args.use_entropy_reg:
                    loss += entropy_loss
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

            batch_loss = loss.item()
            epoch_loss += batch_loss
            if step == 0 and epoch == 1:
                running_avg_loss = batch_loss
            else:
                running_avg_loss = alpha * running_avg_loss + (1 - alpha) * batch_loss

            if step % 10 == 0:
                logging.info(f"[Finetune] Epoch {epoch}, Batch {step + 1}/{total_batches}, "
                             f"Loss: {batch_loss:.4f}, Running Avg Loss: {running_avg_loss:.4f}")
                logging.info(f"Outputs: {output[:5]}, Labels: {labels[:5]}, EBM Energy: {ebm_energy[:5]}")
                logging.info(f"Task Loss: {task_loss.item():.4f}, EBM Loss: {ebm_loss.item():.4f}, "
                             f"Entropy Loss: {entropy_loss.item():.4f}")

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

            if args.use_si and si:
                si.update_weights(model)
            if args.use_replay_buffer and replay_buffer:
                replay_buffer.add_batch(batch)

        avg_epoch_loss = epoch_loss / total_batches
        logging.info(f"[Finetune] Finished Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}")
        if args.use_ewc and ewc:
            for ewc_instance in ewc:
                ewc_instance.consolidate(model)
        if args.use_si and si:
            si.update_omega(model)
        gc.collect()

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Save final fine-tuning checkpoint
    finetune_tag = "final"
    finetune_dir = os.path.join(args.save_dir, finetune_tag)
    os.makedirs(finetune_dir, exist_ok=True)
    try:
        if use_deepspeed:
            model.save_checkpoint(args.save_dir, tag=finetune_tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(finetune_dir, "model.pth"))
    except Exception as e:
        raise RuntimeError(f"Rank {rank} failed to save finetune checkpoint: {str(e)}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if use_deepspeed and rank == 0:
        consolidated_path = os.path.join(args.save_dir, "consolidated_" + finetune_tag + ".pth")
        state_dict = model.module.state_dict()
        torch.save(state_dict, consolidated_path)
        if os.path.exists(consolidated_path):
            logging.info(f"Final consolidated model weights saved to {consolidated_path}")
        else:
            logging.error(f"Failed to save final consolidated weights at {consolidated_path}")
    if (not torch.distributed.is_initialized()) or (rank == 0):
        if args.bucket:
            upload_checkpoint_to_s3(args.save_dir, args.bucket, remote_dir="model")
    logging.info("All epochs completed.")
