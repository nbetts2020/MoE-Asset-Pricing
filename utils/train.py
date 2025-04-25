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
from utils.data import prepare_ft_dataloader
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from deepspeed.runtime.zero.stage3 import GatheredParameters

# Import the helper function to update RoPE buffers.
from utils.model import update_model_rope_for_extended_context, expand_pos_embedding

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _seq_parallel_slice(full_ids, rank, world_size):
    """
    Split (B,T_full,[...]) along the sequence dimension so each rank keeps a
    contiguous slice of length T_full/world_size.  Works for both (B,T) and
    (B,K,T) layouts.
    """
    t_dim = -1                                 # last dim is sequence
    T_full = full_ids.size(t_dim)
    assert T_full % world_size == 0, "T_full must be divisible by world_size"
    T_local = T_full // world_size
    start, end = rank * T_local, (rank + 1) * T_local
    slicer = [slice(None)] * full_ids.dim()
    slicer[t_dim] = slice(start, end)
    ids_local = full_ids[tuple(slicer)]
    return ids_local, start                    # ids_local, global_offset


def extract_label_value(decoded_text):
    """Extracts the numerical label after '<30 DAY LABEL>:'."""
    match = re.search(r'<30 DAY LABEL>:\s*([\d\.]+)', decoded_text)
    if not match:
        return None
    num_str = re.sub(r'\.\.+', '.', match.group(1))
    try:
        return float(num_str)
    except ValueError as e:
        logging.error(f"Label conversion failed: {num_str} ({e})")
        return None


# ────────────────────────────────────────────────────────────────────────────
# main training loop
# ────────────────────────────────────────────────────────────────────────────
def train_model(
    model,
    optimizer,      # Single optimizer for main model
    epochs,
    device,
    dataloader,
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    tokenizer=None,
    use_deepspeed: bool = False,
):
    """
    Training loop with four phases:
      1. Normal pre-training
      2. Continual pre-training with 64 k context
      3. Coconut SFT
      4. EBM fine-tuning + validation
    """
    engine = model if use_deepspeed else None
    adam_optimizer = optimizer if not use_deepspeed else None

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )

    # ----------------------------------------------------------------------
    # PHASE 1 : Normal Pre-training
    # ----------------------------------------------------------------------
    logging.info(f"[Rank {rank}] starting normal pre-training.")
    PRETRAIN_EPOCHS = epochs
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            full_ids = batch["input_ids"].to(device)       # (B, T_full)
            ids, seq_offset = _seq_parallel_slice(full_ids, rank, world_size)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_next_token_efficient(
                        ids, reduction="mean", offset=seq_offset
                    )

            # global average loss
            if world_size > 1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                adam_optimizer.step()

            # regularisers / replay
            if args.use_si and si:
                si.update_weights(model)
            if args.use_ewc and ewc:
                for ewc_inst in ewc:
                    loss = loss + args.lambda_ewc * ewc_inst.penalty(model)
            if replay_buffer and len(replay_buffer.buffer):
                loss = loss + replay_buffer.replay_and_calculate_loss(
                    model, tokenizer, args.replay_batch_size, device, args.replay_buffer_weight
                )

            epoch_loss += loss.item()
        logging.info(
            f"[Normal] Epoch {epoch}/{PRETRAIN_EPOCHS} | "
            f"avg loss: {epoch_loss/len(dataloader):.4f}"
        )

    # checkpoint
    pretrain_tag = "normal_pretrained"
    os.makedirs(os.path.join(args.save_dir, pretrain_tag), exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=pretrain_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, pretrain_tag, "model.pth"))

    if getattr(args, "stage_1_only", False):
        logging.info("stage_1_only=True – exiting after phase 1.")
        return

    # ----------------------------------------------------------------------
    # PHASE 2 : Continual Pre-training (64 k)
    # ----------------------------------------------------------------------
    logging.info("=== Continual pre-training @64 k context ===")
    config.BLOCK_SIZE = config.CONTEXT_WINDOW = 65_536
    model.tokenizer.model_max_length = config.CONTEXT_WINDOW
    expand_pos_embedding(model, new_len=config.BLOCK_SIZE)
    update_model_rope_for_extended_context(model, new_len=config.BLOCK_SIZE, base=5e5)

    from utils.utils import prepare_optimizer
    new_opts = prepare_optimizer(model, args)
    if use_deepspeed:
        engine.optimizer = new_opts["main"]
    else:
        adam_optimizer = new_opts["main"]

    continual_loader = prepare_ft_dataloader(
        tokenizer, block_size=config.BLOCK_SIZE, shuffle=False, args=args, stage=1
    )

    for epoch in range(1, 2):
        model.train()
        epoch_loss = 0.0
        for batch in continual_loader:
            full_ids = batch["input_ids"].to(device)
            ids, seq_offset = _seq_parallel_slice(full_ids, rank, world_size)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_next_token_efficient(
                        ids, reduction="mean", offset=seq_offset
                    )

            if world_size > 1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size

            if use_deepspeed:
                engine.zero_grad(); engine.backward(loss); engine.step()
            else:
                adam_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                adam_optimizer.step()

            epoch_loss += loss.item()
        logging.info(f"[Continual] Epoch 1/1 | avg loss: {epoch_loss/len(continual_loader):.4f}")

    continual_tag = "continual_pretrained_64k"
    os.makedirs(os.path.join(args.save_dir, continual_tag), exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=continual_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, continual_tag, "model.pth"))

    # ----------------------------------------------------------------------
    # PHASE 3 : Coconut SFT
    # ----------------------------------------------------------------------
    logging.info("=== Coconut supervised fine-tune ===")
    config.LEARNING_RATE, config.LR_DECAY, config.DROPOUT = 1e-5, 0.95, 0.05

    for sub_epoch in range(2):
        gradual = sub_epoch == 0
        loader = prepare_ft_dataloader(
            tokenizer,
            block_size=config.BLOCK_SIZE,
            shuffle=True,
            args=args,
            stage=2,
            gradual_latent_mask=gradual,
            full_latent_mask=not gradual,
        )
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            full_ids = batch["input_ids"].to(device)
            ids, seq_offset = _seq_parallel_slice(full_ids, rank, world_size)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_coconut(
                        ids,
                        attention_mask=None,
                        labels=ids,
                        latent_token_id=model.tokenizer.convert_tokens_to_ids("<bot>"),
                        reduction="mean",
                        offset=seq_offset,
                    )
            if world_size > 1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size

            if use_deepspeed:
                engine.zero_grad(); engine.backward(loss); engine.step()
            else:
                adam_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                adam_optimizer.step()

            epoch_loss += loss.item()
        logging.info(
            f"[Coconut] sub-epoch {sub_epoch+1}/2 | avg loss: {epoch_loss/len(loader):.4f}"
        )

    coconut_tag = "supervised_finetuned_coconut"
    os.makedirs(os.path.join(args.save_dir, coconut_tag), exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=coconut_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, coconut_tag, "model.pth"))

    # ----------------------------------------------------------------------
    # PHASE 4 : EBM fine-tune + validation
    # ----------------------------------------------------------------------
    logging.info("=== EBM fine-tuning ===")
    ebm_opt = torch.optim.Adam(model.ebm.parameters(), lr=args.ebm_lr)
    margin = getattr(args, "ebm_margin", 1.0)

    for epoch in range(1, args.ebm_epochs + 1):
        model.ebm.train()
        total_loss, steps = 0.0, 0
        for stage in range(3, 8):
            loader = prepare_ft_dataloader(
                tokenizer, block_size=config.BLOCK_SIZE, shuffle=True, args=args, stage=stage
            )
            for batch in loader:
                ids_3d = batch["input_ids"].to(device)        # (B, K, T_full)
                ids_flat, seq_offset = _seq_parallel_slice(ids_3d, rank, world_size)
                B, K, T_local = ids_flat.shape

                with torch.no_grad():
                    flat_embs = model.get_embeddings(
                        ids_flat.view(B * K, T_local), pool=True, offset=seq_offset
                    )
                embs = flat_embs.view(B, K, -1)
                flat_embs = embs.view(B * K, -1)
                energies = model.ebm(flat_embs).view(B, K)

                pos_idx = energies.argmin(dim=1)
                pos_en = energies[torch.arange(B), pos_idx]
                neg_en = energies.masked_select(
                    ~torch.nn.functional.one_hot(pos_idx, K).bool()
                ).view(B, K - 1)

                loss = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()

                if world_size > 1:
                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                    loss = loss / world_size

                ebm_opt.zero_grad(); loss.backward(); ebm_opt.step()
                total_loss += loss.item(); steps += 1

        logging.info(f"[EBM-FT] epoch {epoch}/{args.ebm_epochs} | avg loss {total_loss/steps:.4f}")

    # validation
    logging.info("=== EBM validation ===")
    model.ebm.eval()
    val_loader = prepare_ft_dataloader(
        tokenizer, block_size=config.BLOCK_SIZE, shuffle=False, args=args, stage=8
    )
    val_loss, steps = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            ids_3d = batch["input_ids"].to(device)
            ids_flat, seq_offset = _seq_parallel_slice(ids_3d, rank, world_size)
            B, K, T_local = ids_flat.shape

            flat_embs = model.get_embeddings(
                ids_flat.view(B * K, T_local), pool=True, offset=seq_offset
            )
            energies = model.ebm(flat_embs).view(B, K)
            pos_idx = energies.argmin(dim=1)
            pos_en = energies[torch.arange(B), pos_idx]
            neg_en = energies.masked_select(
                ~torch.nn.functional.one_hot(pos_idx, K).bool()
            ).view(B, K - 1)

            loss = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
            if world_size > 1:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size

            val_loss += loss.item(); steps += 1
    logging.info(f"[EBM-Val] avg loss: {val_loss/steps:.4f}")

    # final checkpoint
    final_tag = "model_with_ebm"
    os.makedirs(os.path.join(args.save_dir, final_tag), exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=final_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(args.save_dir, final_tag, "model.pth"))

    logging.info("Training complete.")
