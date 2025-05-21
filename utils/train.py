import os
import gc
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
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
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

# Import the helper function to update RoPE buffers.
from utils.model import update_model_rope_for_extended_context

STAGE_TAG = {
    1: "normal_pretrained",
    2: "continual_pretrained_64k",
    3: "supervised_finetuned_coconut",
    4: "model_with_ebm",
}

def extract_label_value(decoded_text):
    """
    Extracts the numerical label between
    '<STOCK PRICE 30 DAYS OUT>:' and '</STOCK PRICE 30 DAYS OUT>'.
    """
    m = re.search(
        r'<STOCK PRICE 30 DAYS OUT>:\s*([\d\.]+)\s*<\/STOCK PRICE 30 DAYS OUT>',
        decoded_text
    )
    if not m:
        return None
    num = re.sub(r'\.\.+', '.', m.group(1))
    try:
        return float(num)
    except ValueError:
        logging.error(f"Bad float '{num}' in \"{decoded_text}\"")
        return None

def greedy_generate_tail(model, input_ids, max_new_tokens=20):
    """
    Greedy-generate up to `max_new_tokens` *without* extending RoPE.
    Only the last T_local tokens of the prefix are fed to each rank.
    Returns the newly generated ids (B × T_new).
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    T_local    = model.block_size // world_size

    cur = input_ids[:, -T_local:].clone()          # keep last T_local tokens
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            hs     = model.forward_embeddings_only(cur)      # (B, t, C)
            logits = model.lm_head(hs)                       # (B, t, V)
            nxt    = logits[:, -1, :].argmax(-1, keepdim=True)  # (B,1)
            generated.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
            cur = cur[:, -T_local:]          # always trim to T_local
    return torch.cat(generated, dim=1)       # (B, max_new_tokens)

@torch.no_grad()
def local_logits(model, ids_local, mask_local):
    x = model.token_embedding_table(ids_local)
    x = model.dropout_emb(x)
    for blk in model.blocks:
        x, _ = blk(x, mask=mask_local)      # ignore MoE aux‑loss
    x = model.ln_f(x)
    return model.lm_head(x)                 # (B, T_local, vocab)

def train_model(
    model,
    optimizer,      # single optimizer for main model
    epochs,
    device,
    dataloader,
    args,
    si=None,
    ewc=None,
    replay_buffer=None,
    tokenizer=None,
    use_deepspeed=False,
):
    """
    Training loop:
      1. Normal pre-training          (ft_dataset_1)
      2. Continual pre-training 64 k   (ft_dataset_1)
      3. Supervised fine-tuning        (ft_dataset_2)
      4. EBM fine-tuning + validation  (ft_dataset_3-8)
    """

    # --------------------------------------------------------------------- #
    #  Pick the *real* model object (engine.module when using DeepSpeed)
    # --------------------------------------------------------------------- #
    if use_deepspeed:
        engine      = model
        real_model  = engine.module
    else:
        engine      = None
        adam_optimizer = optimizer
        real_model  = model

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank       = dist.get_rank()       if dist.is_initialized() else 0

    # --------------------------------------------------------------------- #
    #  Helper: pad (or left-trim) to the model’s *current* global context
    # --------------------------------------------------------------------- #
    def pad_to_global(ids: torch.Tensor) -> torch.Tensor:
        """
        Pad / trim *on the left* so every sample is exactly
        real_model.block_size tokens long, then return that full-length
        tensor (sharding happens later inside forward).
        """
        B, T = ids.shape
        T_full    = real_model.block_size           # single source of truth
        pad_id    = real_model.tokenizer.pad_token_id

        if T < T_full:
            pad = torch.full(
                (B, T_full - T), pad_id,
                dtype=ids.dtype, device=ids.device
            )
            full = torch.cat([ids, pad], dim=1)
        else:
            full = ids[:, -T_full:]                 # trim from the left

        return full
    # --------------------------------------------------------------------- #

    # -----------------------------------------------------------
    #  If we start at stage > 1, restore the previous checkpoint
    # -----------------------------------------------------------
    min_stage = min(args.stages)
    if min_stage > 1:
        prev_tag = STAGE_TAG[min_stage - 1]
        ckpt_dir = os.path.join(args.save_dir, prev_tag)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(
                f"Cannot start at stage {min_stage}: checkpoint '{prev_tag}' "
                f"not found in {ckpt_dir}"
            )

        if engine:
            engine.load_checkpoint(args.save_dir, tag=prev_tag)
        else:
            path = os.path.join(ckpt_dir, "model.pth")
            real_model.load_state_dict(torch.load(path, map_location="cpu"))
        logging.info(f"Loaded checkpoint from previous stage '{prev_tag}'")

    # ===================================================================== #
    #  PHASE 1 – Normal pre-training
    # ===================================================================== #
    if 1 in args.stages:
        logging.info(f"→ Phase 1: normal pre-training (rank {rank})")
        PRETRAIN_EPOCHS = epochs

        open_ids = tokenizer("<STOCK PRICE 30 DAYS OUT>: ", add_special_tokens=False).input_ids
        m_len    = len(open_ids)
        close_id = tokenizer.convert_tokens_to_ids(" </STOCK PRICE 30 DAYS OUT>")

        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            real_model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                print(step, len(dataloader), "progress!!")
                # -------- quick sanity print every 150 mini-batches ----------
                if (step + 1) % 150 == 0:
                    with torch.no_grad():
                        sample_ids  = batch["input_ids"][:5].to(device)
                        true_prices = batch["labels"][:5].tolist()
                        preds = []
                
                        for ids in sample_ids:
                            ids_list = ids.tolist()
                            marker_pos = next(
                                (j + m_len for j in range(len(ids_list) - m_len + 1)
                                 if ids_list[j:j + m_len] == open_ids),
                                None,
                            )
                            if marker_pos is None:
                                preds.append("〈n/a〉")
                                continue
                
                            # greedy‐generate until the closing tag appears
                            prefix     = ids[:marker_pos].unsqueeze(0)
                            world_size = dist.get_world_size() if dist.is_initialized() else 1
                            T_local    = real_model.block_size // world_size
                            cur        = prefix[:, -T_local:].clone()
                            gen_tokens = []
                
                            for _ in range(20):
                                hs     = real_model.forward_embeddings_only(cur)
                                logits = real_model.lm_head(hs)[:, -1, :]
                                nxt    = logits.argmax(-1, keepdim=True)
                                gen_tokens.append(nxt)
                                if nxt.item() == close_id:
                                    break
                                cur = torch.cat([cur, nxt], dim=1)[:, -T_local:]
                
                            gen_ids  = torch.cat(gen_tokens, dim=1)
                            full_ids = torch.cat([prefix, gen_ids], dim=1)[0]
                            decoded  = tokenizer.decode(full_ids, skip_special_tokens=True)
                            val      = extract_label_value(decoded)
                            preds.append(f"{val:.2f}" if val is not None else "〈n/a〉")
                
                        if rank == 0:
                            for i, pred in enumerate(preds):
                                print(f"[b{step+1} s{i}] true → {true_prices[i]:.2f}, pred → {pred}")

                # ------------------------------------------------------------

                input_ids = pad_to_global(batch["input_ids"].to(device))
                loss = real_model.forward_next_token_efficient(
                    input_ids, reduction="mean"
                )

                if engine:
                    engine.zero_grad(); engine.backward(loss); engine.step()
                else:
                    adam_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
                    adam_optimizer.step()

                # regularisers / replay
                if args.use_si and si:
                    si.update_weights(real_model)
                if args.use_ewc and ewc:
                    for ewc_inst in ewc:
                        loss = loss + args.lambda_ewc * ewc_inst.penalty(real_model)
                if replay_buffer and len(replay_buffer.buffer):
                    loss += replay_buffer.replay_and_calculate_loss(
                        model=real_model,
                        tokenizer=tokenizer,
                        replay_batch_size=args.replay_batch_size,
                        device=device,
                        alpha=args.replay_buffer_weight,
                    )

                epoch_loss += loss.item()

            logging.info(
                f"[Phase 1] Epoch {epoch}/{PRETRAIN_EPOCHS} — "
                f"avg loss {epoch_loss/len(dataloader):.4f}"
            )

        # save after phase 1
        tag = STAGE_TAG[1]
        if engine:
            engine.save_checkpoint(args.save_dir, tag=tag)
        else:
            os.makedirs(os.path.join(args.save_dir, tag), exist_ok=True)
            torch.save(
                real_model.state_dict(),
                os.path.join(args.save_dir, tag, "model.pth"),
            )

        if getattr(args, "stage_1_only", False):
            logging.info("stage_1_only=True → stopping after phase 1")
            return

    # ===================================================================== #
    #  PHASE 2 – Continual pre-training with extended context
    # ===================================================================== #
    if 2 in args.stages:
        logging.info("→ Phase 2: continual pre-training (64 k context)")

        NEW_LEN = 2048
        # one call updates RoPE *and* sets real_model.block_size/tokenizer
        update_model_rope_for_extended_context(real_model, new_seq_len=NEW_LEN,
                                               base=500_000.0)

        # new optimiser (LR schedule, new parameters, …)
        from utils.utils import prepare_optimizer
        new_opts = prepare_optimizer(real_model, args)
        if engine:
            logging.info("DeepSpeed engine retains its optimizer")
        else:
            adam_optimizer = new_opts["main"]

        continual_loader = prepare_ft_dataloader(
            tokenizer, block_size=real_model.block_size,
            shuffle=False, args=args, stage=1
        )

        real_model.train()
        epoch_loss = 0.0
        for batch in continual_loader:
            input_ids = pad_to_global(batch["input_ids"].to(device))
            loss = real_model.forward_next_token_efficient(
                input_ids, reduction="mean"
            )

            if engine:
                engine.zero_grad(); engine.backward(loss); engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
                adam_optimizer.step()

            epoch_loss += loss.item()

        logging.info(f"[Phase 2] Avg loss {epoch_loss/len(continual_loader):.4f}")

        tag = STAGE_TAG[2]
        if engine:
            engine.save_checkpoint(args.save_dir, tag=tag)
        else:
            os.makedirs(os.path.join(args.save_dir, tag), exist_ok=True)
            torch.save(
                real_model.state_dict(),
                os.path.join(args.save_dir, tag, "model.pth"),
            )

    # ---------------------------------------------------------------------#
    #  PHASE 3 – Supervised Coconut fine-tuning with N-stage mask schedule #
    # ---------------------------------------------------------------------#
    if 3 in args.stages:
        logging.info("=== Starting Supervised Fine-Tuning (Coconut) ===")
        config.LEARNING_RATE = 1e-5
        config.LR_DECAY      = 0.95
    
        # Build ONE dataloader (the parquet is read only once)
        ft_loader = prepare_ft_dataloader(
            tokenizer   = tokenizer,
            block_size  = config.BLOCK_SIZE,
            shuffle     = True,
            args        = args,
            stage       = 2,               # ft_dataset_2
        )
        ds: PrecomputedDataset = ft_loader.dataset        # type: ignore
    
        # Iterate over the mask curriculum
        mask_stages = getattr(config, "COCONUT_MASK_STAGES", [0.0, 1.0])
        for stage_idx, frac in enumerate(mask_stages):
            ds.mask_fraction = float(frac)        # schedule_masking will use this
            logging.info(
                f"→ Coconut stage {stage_idx+1}/{len(mask_stages)} "
                f"— mask {frac*100:.0f}% of reasoning tokens"
            )
    
            real_model.train()
            epoch_loss = 0.0
    
            for step, batch in enumerate(ft_loader):
                input_ids = pad_to_global(batch["input_ids"].to(device))
    
                loss = real_model.forward_coconut(
                    input_ids       = input_ids,
                    attention_mask  = None,
                    labels          = input_ids,
                    latent_token_id = model.tokenizer.convert_tokens_to_ids("<bot>"),
                    reduction       = "mean",
                )
    
                if use_deepspeed:
                    engine.zero_grad()
                    engine.backward(loss)
                    engine.step()
                else:
                    adam_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
                    adam_optimizer.step()
    
                epoch_loss += loss.item()
    
            logging.info(
                f"[Coconut] Stage {stage_idx+1}/{len(mask_stages)}, "
                f"avg loss {epoch_loss/len(ft_loader):.4f}"
            )
    
        # Save checkpoint after the full curriculum
        coconut_tag  = "supervised_finetuned_coconut"
        coconut_dir  = os.path.join(args.save_dir, coconut_tag)
        os.makedirs(coconut_dir, exist_ok=True)
        if use_deepspeed:
            engine.save_checkpoint(args.save_dir, tag=coconut_tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(coconut_dir, "model.pth"))

    # ------------------------------------------------------------------------
    # PHASE 4: EBM Fine-Tuning (ft_dataset_3-7) + Validation (ft_dataset_8)
    # ------------------------------------------------------------------------
    if 4 in args.stages:
        logging.info("=== Starting EBM Fine-Tuning Phase ===")
        ebm_optimizer = torch.optim.Adam(model.ebm.parameters(), lr=args.ebm_lr)
        margin = getattr(args, "ebm_margin", 1.0)
        ebm_epochs = 1
        for epoch in range(1, ebm_epochs + 1):
            real_model.ebm.train()
            total_loss = 0.0
            steps = 0
    
            for stage in range(3, 8):
                ebm_loader = prepare_ft_dataloader(
                    tokenizer,
                    block_size=config.BLOCK_SIZE,
                    shuffle=True,
                    args=args,
                    stage=stage
                )
                for batch in ebm_loader:
                    ids = batch['input_ids'].to(device)
                    ids = pad_to_global(input_ids)
                    B, K, T = ids.shape
                    flat_ids = ids.view(B*K, T)
    
                    with torch.no_grad():
                        embs = real_model.get_embeddings(flat_ids, pool=True)
                    embs = embs.view(B, K, -1)
    
                    flat_embs = embs.view(B*K, -1)
                    energies = real_model.ebm(flat_embs).view(B, K)
    
                    pos_idx = energies.argmin(dim=1)
                    pos_en = energies[torch.arange(B), pos_idx]
                    neg_mask = torch.ones_like(energies, dtype=torch.bool)
                    neg_mask[torch.arange(B), pos_idx] = False
                    neg_en = energies[neg_mask].view(B, K-1)
    
                    loss = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
                    ebm_optimizer.zero_grad()
                    loss.backward()
                    ebm_optimizer.step()
    
                    total_loss += loss.item()
                    steps += 1
    
            logging.info(f"[EBM FT] Epoch {epoch}/{args.ebm_epochs}, Avg Loss: {total_loss/steps:.4f}")
    
        # validation on ft_dataset_8
        logging.info("=== EBM Validation ===")
        real_model.ebm.eval()
        val_loader = prepare_ft_dataloader(
            tokenizer,
            block_size=config.BLOCK_SIZE,
            shuffle=False,
            args=args,
            stage=8
        )
        val_loss = 0.0
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                ids = pad_to_global(ids)
                B, K, T = ids.shape
                flat_ids = ids.view(B*K, T)
                embs = real_model.get_embeddings(flat_ids, pool=True).view(B, K, -1)
                energies = real_model.ebm(embs.view(B*K, -1)).view(B, K)
                pos_idx = energies.argmin(dim=1)
                pos_en = energies[torch.arange(B), pos_idx]
                neg_mask = torch.ones_like(energies, dtype=torch.bool)
                neg_mask[torch.arange(B), pos_idx] = False
                neg_en = energies[neg_mask].view(B, K-1)
                loss = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
                val_loss += loss.item()
                steps += 1
        logging.info(f"[EBM Val] Avg Loss: {val_loss/steps:.4f}")
    
        # save final model+EBM
        final_tag = "model_with_ebm"
        final_dir = os.path.join(args.save_dir, final_tag)
        os.makedirs(final_dir, exist_ok=True)
        if use_deepspeed:
            engine.save_checkpoint(args.save_dir, tag=final_tag, client_state={})
        else:
            torch.save(model.state_dict(), os.path.join(final_dir, "model_with_ebm.pth"))
    
        logging.info("Training complete.")
