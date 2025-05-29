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
import deepspeed

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

def dbg(msg: str):
    """Print from **all** ranks and flush right away."""
    r = dist.get_rank() if dist.is_initialized() else 0
    print(f"[rank {r}] {msg}", flush=True)

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
        prev_tag  = STAGE_TAG[min_stage - 1]               # e.g. "normal_pretrained"
        ckpt_root = args.save_dir
        stage_dir = os.path.join(ckpt_root, prev_tag)

        # ── rank-0 downloads just the ZeRO shard files ────────────
        if rank == 0 and not os.path.isdir(stage_dir):
            os.makedirs(stage_dir, exist_ok=True)
            subprocess.check_call([
                "aws", "s3", "sync", "--region", "us-west-1",
                f"s3://{args.bucket}/model/{prev_tag}/", stage_dir
            ])
        if world_size > 1:
            dist.barrier()

        # ── load *only* the module weights ────────────────────────────
        if engine:
            engine.load_checkpoint(
                load_dir=stage_dir,           # ← root of all stage-dirs
                tag=None,                 # ← name of the subfolder you just synced
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_only=True,
                load_module_strict=False
            )
        else:
            real_model.load_state_dict(
                torch.load(
                    os.path.join(
                        stage_dir,
                        "zero_pp_rank_0_mp_rank_00_model_states.pt"
                    ),
                    map_location="cpu"
                ),
                strict=True
            )

        logging.info(f"Loaded checkpoint from Phase {min_stage} ('{prev_tag}')")
        
    real_model.ebm = real_model.ebm.to(torch.bfloat16)
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
        if not torch.distributed.is_initialized() or dist.get_rank() == 0:
            upload_checkpoint_to_s3(
                local_dir=os.path.join(args.save_dir, STAGE_TAG[1]),
                bucket=args.bucket,
                remote_dir=f"model/{STAGE_TAG[1]}"
            )

        if getattr(args, "stage_1_only", False):
            logging.info("stage_1_only=True → stopping after phase 1")
            return

    # ===================================================================== #
    #  PHASE 2 – Continual pre-training with extended context
    # ===================================================================== #
    if 2 in args.stages:
        dbg("→ entering Phase 2")

        # ── 0. load the Phase-1 checkpoint before touching RoPE or SP groups ─────
        dbg("loading Phase-1 checkpoint")
        engine.load_checkpoint(args.save_dir, tag=STAGE_TAG[1])
        real_model = engine.module  # unwrap DeepSpeed
        dbg("loaded Phase-1 model")

        NEW_LEN = 4096                     # target context length
        device   = torch.device("cuda")

        # ── 1. global sync so all ranks have the same model state ───────────
        dbg("before barrier #1")
        dist.barrier()
        dbg("passed barrier #1")
        real_model.block_size = NEW_LEN
        config.BLOCK_SIZE = NEW_LEN
        # ── 2. extend the RoPE tables in-place ──────────────────────────────
        dbg("calling update_model_rope_for_extended_context")
        update_model_rope_for_extended_context(
            real_model,
            new_seq_len = NEW_LEN,
            base        = 10_000.0,
        )
        dbg(f"after RoPE update: block_size = {real_model.block_size}")

        # ── 3. rebuild sequence-parallel process groups & each attention ────
        if dist.is_initialized():
            from yunchang import set_seq_parallel_pg, LongContextAttention
            from yunchang.kernels import AttnType

            set_seq_parallel_pg(
                config.SP_ULYSSES_DEGREE,
                config.SP_RING_DEGREE,
                dist.get_rank(),
                dist.get_world_size(),
            )

            for blk in real_model.blocks:
                blk.sa.usp_attn = LongContextAttention(
                    scatter_idx     = 2,
                    gather_idx      = 1,
                    ring_impl_type  = "zigzag",
                    use_pack_qkv    = True,
                    use_sync        = True,
                    attn_type       = AttnType.FA,
                    attn_processor  = None,
                )

        # ── 4. zero out any optimizer state so momentum/etc doesn’t mix old context ─
        dbg("zeroing DeepSpeed optimizer state")
        engine.optimizer.zero_grad(set_to_none=True)

        # ── 5. sync again before you start training on the new context ─────────
        dbg("before barrier #2")
        dist.barrier()
        dbg("passed barrier #2")

        # ── 6. tiny RoPE-FT pass ─────────────────────────────────────────────
        dbg("building continual_loader")
        continual_loader = prepare_ft_dataloader(
            tokenizer,
            block_size = NEW_LEN,
            shuffle    = False,
            args       = args,
            stage      = 1,
            streaming  = True,
        )
        dbg("continual_loader ready")

        real_model.train()
        epoch_loss = 0.0
        dbg("entering training loop")

        for idx, batch in enumerate(continual_loader):
            print(idx, len(continual_loader))
            bsz = batch["input_ids"].size(0)
            if bsz != config.BATCH_SIZE:
                dbg(f"Skipping stray batch of size {bsz}")
                continue

            input_ids = pad_to_global(batch["input_ids"].to(device))
            loss      = real_model.forward_next_token_efficient(
                            input_ids, reduction="mean"
                        )
            print(loss, "loss!!")

            engine.zero_grad()
            engine.backward(loss)
            engine.step()

            epoch_loss += loss.item()

        logging.info(f"[Phase 2] Avg loss {epoch_loss/len(continual_loader):.4f}")

        # ── 7. save & upload the Phase-2 checkpoint ───────────────────────────
        tag = STAGE_TAG[2]
        engine.save_checkpoint(args.save_dir, tag=tag)

        if not dist.is_initialized() or dist.get_rank() == 0:
            upload_checkpoint_to_s3(
                local_dir  = os.path.join(args.save_dir, tag),
                bucket     = args.bucket,
                remote_dir = f"model/{tag}",
            )


    # --------------------------------------------------------------------- #
    #  PHASE 3 – Coconut (supervised) fine-tuning with mask curriculum      #
    # --------------------------------------------------------------------- #
    if 3 in args.stages:
        dbg("→ entering Phase 3")

        # ── 0. load the Phase-1 checkpoint before touching RoPE or SP groups ─────
        dbg("loading Phase-2 checkpoint")
        engine.load_checkpoint(args.save_dir, tag=STAGE_TAG[2])
        real_model = engine.module  # unwrap DeepSpeed
        dbg("loaded Phase-2 model")

        device   = torch.device("cuda")

        # ── 3. rebuild sequence-parallel process groups & each attention ────
        if dist.is_initialized():
            from yunchang import set_seq_parallel_pg, LongContextAttention
            from yunchang.kernels import AttnType

            set_seq_parallel_pg(
                config.SP_ULYSSES_DEGREE,
                config.SP_RING_DEGREE,
                dist.get_rank(),
                dist.get_world_size(),
            )

            for blk in real_model.blocks:
                blk.sa.usp_attn = LongContextAttention(
                    scatter_idx     = 2,
                    gather_idx      = 1,
                    ring_impl_type  = "zigzag",
                    use_pack_qkv    = True,
                    use_sync        = True,
                    attn_type       = AttnType.FA,
                    attn_processor  = None,
                )

        # ── 4. zero out any optimizer state so momentum/etc doesn’t mix old context ─
        dbg("zeroing DeepSpeed optimizer state")
        engine.optimizer.zero_grad(set_to_none=True)

        # ── 5. sync again before you start training on the new context ─────────
        dbg("before barrier #2")
        dist.barrier()
        dbg("passed barrier #2")

        # ── 6. tiny RoPE-FT pass ─────────────────────────────────────────────
        dbg("building continual_loader")
        continual_loader = prepare_ft_dataloader(
            tokenizer,
            block_size = config.BLOCK_SIZE,
            shuffle    = False,
            args       = args,
            stage      = 2,
            streaming  = True,
        )
        dbg("continual_loader ready")

        real_model.train()
        epoch_loss = 0.0
        dbg("entering training loop")

        for idx, batch in enumerate(continual_loader):
            print(idx, len(continual_loader))
            bsz = batch["input_ids"].size(0)
            if bsz != config.BATCH_SIZE:
                dbg(f"Skipping stray batch of size {bsz}")
                continue

            input_ids = pad_to_global(batch["input_ids"].to(device))
            loss      = real_model.forward_next_token_efficient(
                            input_ids, reduction="mean"
                        )
            print(loss, "loss!!")

            engine.zero_grad()
            engine.backward(loss)
            engine.step()

            epoch_loss += loss.item()

        logging.info(f"[Phase 3] Avg loss {epoch_loss/len(continual_loader):.4f}")

        # ── 7. save & upload the Phase-2 checkpoint ───────────────────────────
        tag = STAGE_TAG[3]
        engine.save_checkpoint(args.save_dir, tag=tag)

        if not dist.is_initialized() or dist.get_rank() == 0:
            upload_checkpoint_to_s3(
                local_dir  = os.path.join(args.save_dir, tag),
                bucket     = args.bucket,
                remote_dir = f"model/{tag}",
            )
    # if 3 in args.stages:

    #     logging.info("=== Starting Supervised Fine-Tuning (Coconut) ===")
    #     config.LEARNING_RATE = 1e-5 # Set learning rate for this stage
    #     config.LR_DECAY      = 0.95 # Set learning rate decay for this stage

    #     mask_stages = config.COCONUT_MASK_STAGES # Get mask stages from config
    #     mask_stages = [0.05, 1.0] # Example override for testing

    #     for stage_idx, frac in enumerate(mask_stages):
    #         logging.info(
    #             f"→ Coconut curriculum stage {stage_idx+1}/{len(mask_stages)}  — "
    #             f"mask_fraction {frac*100:.0f}%"
    #         )

    #         # Re-create the dataloader for each mask fraction stage to ensure it's applied
    #         ft_loader = prepare_ft_dataloader(
    #             tokenizer   = tokenizer, # Pass tokenizer
    #             block_size  = config.BLOCK_SIZE, # Pass block size
    #             shuffle     = True, # Shuffle data for training
    #             args        = args, # Pass command line arguments
    #             stage       = 2,    # Use ft_dataset_2.parquet for this phase
    #             streaming   = True, # Use streaming for large datasets
    #             mask_fraction = float(frac) # Pass the current mask fraction to the dataloader
    #         )
    #         # ds: PrecomputedDataset = ft_loader.dataset # No longer need to directly manipulate ds.mask_fraction

    #         real_model.train() # Set model to training mode
    #         epoch_loss = 0.0 # Initialize epoch loss

    #         # Iterate over batches from the dataloader
    #         for step, batch in enumerate(ft_loader):
    #             input_ids = pad_to_global(batch["input_ids"].to(device)) # Pad batch and move to device

    #             # Perform forward pass with Coconut logic
    #             loss = real_model.forward_coconut(
    #                 input_ids       = input_ids,
    #                 attention_mask  = None, # Assuming full attention or handled internally
    #                 labels          = input_ids, # Using input_ids as labels for MLM-like objective or internal handling
    #                 latent_token_id = model.tokenizer.convert_tokens_to_ids("<bot>"), # Pass latent token ID
    #                 reduction       = "mean", # Loss reduction method
    #             )

    #             # Backward pass and optimizer step
    #             if use_deepspeed:
    #                 engine.zero_grad()
    #                 engine.backward(loss)
    #                 engine.step()
    #             else:
    #                 adam_optimizer.zero_grad()
    #                 loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0) # Gradient clipping
    #                 adam_optimizer.step()

    #             epoch_loss += loss.item() # Accumulate loss

    #         logging.info(
    #             f"[Coconut] Curriculum Stage {stage_idx+1}/{len(mask_stages)} — "
    #             f"Avg Epoch Loss: {epoch_loss/len(ft_loader):.4f}"
    #         )

    #     # Save checkpoint after completing all mask stages
    #     coconut_tag = "supervised_finetuned_coconut"
    #     coconut_dir = os.path.join(args.save_dir, coconut_tag)
    #     os.makedirs(coconut_dir, exist_ok=True) # Create directory if it doesn't exist

    #     if use_deepspeed:
    #         engine.save_checkpoint(args.save_dir, tag=coconut_tag, client_state={}) # Save DeepSpeed checkpoint
    #     else:
    #         torch.save(model.state_dict(), os.path.join(coconut_dir, "model.pth")) # Save standard PyTorch model
        
    #     # Upload checkpoint to S3 if applicable
    #     if not torch.distributed.is_initialized() or dist.get_rank() == 0:
    #         upload_checkpoint_to_s3(
    #             local_dir=os.path.join(args.save_dir, STAGE_TAG[3]), # Use STAGE_TAG for consistency
    #             bucket=args.bucket,
    #             remote_dir=f"model/{STAGE_TAG[3]}"
    #         )
            
    # ------------------------------------------------------------------------
    # PHASE 4: EBM Fine-Tuning (ft_dataset_3-7) + Validation (ft_dataset_8)
    # ------------------------------------------------------------------------
    if 4 in args.stages:
        logging.info("=== Starting EBM Fine-Tuning Phase ===")
    
        ebm_optimizer = torch.optim.Adam(real_model.ebm.parameters(), lr=args.ebm_lr)
        margin        = getattr(args, "ebm_margin", 1.0)
        ebm_epochs    = getattr(args, "ebm_epochs", 1)
    
        fw_count = 0  # total forwards seen so far
    
        for epoch in range(1, ebm_epochs + 1):
            real_model.ebm.train()
            total_loss = 0.0
            steps      = 0
    
            for stage in range(3, 8):
                loader = prepare_ft_dataloader(
                    tokenizer,
                    block_size = config.BLOCK_SIZE,
                    shuffle    = True,
                    args       = args,
                    stage      = stage,
                    streaming  = True
                )
    
                for idx, batch in enumerate(loader):
                    print(idx, len(loader), "progress!! 4")
                    ids        = batch["input_ids"].to(device)   # (B, K, T)
                    true_price = batch["labels"].to(device)      # (B,)
    
                    B, K, T = ids.size()
    
                    # 1) Embedding forward for B*K sequences
                    flat_ids = ids.view(B * K, T)
                    with torch.no_grad():
                        embs = real_model.get_embeddings(flat_ids, pool=True)  # (B*K, D)
                    fw_count += B * K
                    print(fw_count, "fw_count1")
    
                    embs = embs.view(B, K, -1)  # (B, K, D)
    
                    # 2) Greedy-generation forwards
                    preds = []
                    for b in range(B):
                        batch_preds = []
                        for k in range(K):
                            gen_ids = greedy_generate_tail(
                                real_model,
                                ids[b, k : k+1, :],  # (1, T)
                                max_new_tokens=20
                            )[0]  # (T_new,)
    
                            fw_count += gen_ids.size(0)
                            print(fw_count, "fw_count2")
    
                            text = tokenizer.decode(
                                torch.cat([ids[b, k : k+1, :], gen_ids.unsqueeze(0)], dim=1)[0],
                                skip_special_tokens=True
                            )
                            val = extract_label_value(text)
                            batch_preds.append(val if val is not None else float("inf"))
                        preds.append(batch_preds)
                    preds = torch.tensor(preds, device=device)  # (B, K)
    
                    # 3) Compute residuals & find pos_idx
                    true_exp = true_price.unsqueeze(1).expand_as(preds)
                    residuals = torch.abs(preds - true_exp)  # (B, K)
                    pos_idx   = residuals.argmin(dim=1)      # (B,)
    
                    # 4) EBM scoring forward for B*K embeddings
                    flat_embs = embs.view(B * K, -1)
                    energies  = real_model.ebm(flat_embs).view(B, K)  # (B, K)
                    fw_count += B * K
                    print(fw_count, "fw_count3")
    
                    # 5) Margin loss
                    pos_en = energies[torch.arange(B), pos_idx]
                    mask   = torch.ones_like(energies, dtype=torch.bool)
                    mask[torch.arange(B), pos_idx] = False
                    neg_en = energies[mask].view(B, K-1)
                    loss   = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
    
                    ebm_optimizer.zero_grad()
                    loss.backward()
                    ebm_optimizer.step()
    
                    total_loss += loss.item()
                    steps      += 1
    
            logging.info(f"[EBM FT] Epoch {epoch}/{ebm_epochs} — Avg Loss {total_loss/steps:.4f}")
    
        # --- Validation (same forward-counter logic, without backward) --- #
        logging.info("=== EBM Validation ===")
        real_model.ebm.eval()
        val_loader = prepare_ft_dataloader(
            tokenizer,
            block_size = config.BLOCK_SIZE,
            shuffle    = False,
            args       = args,
            stage      = 8,
            streaming  = True
        )
        val_loss = 0.0
        steps    = 0
    
        with torch.no_grad():
            for batch in val_loader:
                ids        = batch["input_ids"].to(device)   # (B, K, T)
                true_price = batch["labels"].to(device)      # (B,)
    
                B, K, T = ids.size()
                flat_ids = ids.view(B * K, T)
                embs     = real_model.get_embeddings(flat_ids, pool=True).view(B, K, -1)
    
                fw_count += B * K
                if fw_count % 100 == 0:
                    print(f"[Phase 4] Forward count: {fw_count}")
    
                preds = []
                for b in range(B):
                    batch_preds = []
                    for k in range(K):
                        gen_ids = greedy_generate_tail(
                            real_model,
                            ids[b, k : k+1, :],
                            max_new_tokens=20
                        )[0]
                        fw_count += gen_ids.size(0)
                        if fw_count % 100 == 0:
                            print(f"[Phase 4] Forward count: {fw_count}")
    
                        text = tokenizer.decode(
                            torch.cat([ids[b, k : k+1, :], gen_ids.unsqueeze(0)], dim=1)[0],
                            skip_special_tokens=True
                        )
                        val = extract_label_value(text)
                        batch_preds.append(val if val is not None else float("inf"))
                    preds.append(batch_preds)
                preds     = torch.tensor(preds, device=device)
                residuals = torch.abs(preds - true_price.unsqueeze(1))
                pos_idx   = residuals.argmin(dim=1)
    
                energies = real_model.ebm(embs.view(B*K, -1)).view(B, K)
                fw_count += B * K
                if fw_count % 100 == 0:
                    print(f"[Phase 4] Forward count: {fw_count}")
    
                pos_en = energies[torch.arange(B), pos_idx]
                mask   = torch.ones_like(energies, dtype=torch.bool)
                mask[torch.arange(B), pos_idx] = False
                neg_en = energies[mask].view(B, K-1)
                loss   = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
    
                val_loss += loss.item()
                steps    += 1
    
        logging.info(f"[EBM Val] Avg Loss {val_loss/steps:.4f}")
    
        # --- Save & upload final checkpoint --- #
        final_tag = "model_with_ebm"
        final_dir = os.path.join(args.save_dir, final_tag)
        os.makedirs(final_dir, exist_ok=True)
    
        if use_deepspeed:
            engine.save_checkpoint(args.save_dir, tag=final_tag, client_state={})
        else:
            torch.save(real_model.state_dict(), os.path.join(final_dir, "model_with_ebm.pth"))
    
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            upload_checkpoint_to_s3(
                local_dir  = final_dir,
                bucket     = args.bucket,
                remote_dir = f"model/{final_tag}"
            )
    
        logging.info("=== Phase 4 complete: EBM fine-tuned and saved ===")

        logging.info("Training complete.")
