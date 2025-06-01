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

def save_checkpoint(*, engine, model, save_dir: str, tag: str,
                    bucket: str | None = None):
    """
    Save `model` (or `engine.module`) under save_dir/<tag>/.

    Works for:
      • DeepSpeed ZeRO-3 (engine!=None) – uses engine.save_checkpoint  
      • Plain PyTorch            – torch.save(model.state_dict(), …)

    If `bucket` is given and we’re on rank-0, also sync to S3.
    """
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    if engine is not None:                       # DeepSpeed path
        if rank0:
            print(f"[save] DeepSpeed checkpoint → {save_dir} (tag={tag})")
        engine.save_checkpoint(save_dir, tag=tag)

    else:                                        # plain torch path
        ckpt_dir = os.path.join(save_dir, tag)
        if rank0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pth"))

    if bucket and rank0:
        upload_checkpoint_to_s3(
            local_dir=os.path.join(save_dir, tag),
            bucket=bucket,
            remote_dir=f"model/{tag}",
        )

@torch.no_grad()
def quick_eval_sample(model, tokenizer, batch, *,
                      device, open_ids, close_id, m_len):
    """
    Greedy-generate one mini-batch **on a single GPU** (no NCCL, no ZeRO
    gathers).  Returns a list of (true_price, pred_string).
    """
    model.eval()

    sample_ids  = batch["input_ids"][:5].to(device)
    true_prices = batch["labels"][:5].tolist()
    preds       = []

    for ids in sample_ids:
        ids_list = ids.tolist()
        marker   = next(
            (j + m_len for j in range(len(ids_list) - m_len + 1)
             if ids_list[j : j + m_len] == open_ids),
            None,
        )
        if marker is None:
            preds.append("〈n/a〉"); continue

        prefix   = ids[:marker].unsqueeze(0)          # (1, L)
        cur      = prefix[:, -model.block_size:]      # local-only path
        gen      = []

        while len(gen) < 20:
            hs  = model.forward_embeddings_only(cur,
                                                gather_full=False,
                                                local_only = True)
            nxt = model.lm_head(hs)[:, -1, :].argmax(-1, keepdim=True)
            gen.append(nxt)
            if nxt.item() == close_id: break
            cur = torch.cat([cur, nxt], 1)[:, -model.block_size:]

        full   = torch.cat([prefix, torch.cat(gen, 1)], 1)[0]
        decoded= tokenizer.decode(full, skip_special_tokens=False)
        val    = extract_label_value(decoded)          # your util
        preds.append(f"{val:.2f}" if val is not None else "〈n/a〉")

    model.train()
    return list(zip(true_prices, preds))


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
    #  PHASE 1 – Normal pre-training (updated)                              #
    # ===================================================================== #
    if 1 in args.stages:
        PRETRAIN_EPOCHS = epochs
        logging.info(f"→ Phase 1: normal pre-training for {PRETRAIN_EPOCHS} epoch(s)")
    
        open_ids = tokenizer("<STOCK PRICE 30 DAYS OUT>: ", add_special_tokens=False).input_ids
        m_len    = len(open_ids)
        close_id = tokenizer.convert_tokens_to_ids("</STOCK PRICE 30 DAYS OUT>")
    
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            real_model.train()
            epoch_loss = 0.0
    
            # ————————————— mini-batch loop —————————————
            for step, batch in enumerate(dataloader):
                input_ids = pad_to_global(batch["input_ids"].to(device))
                loss      = real_model.forward_next_token_efficient(input_ids, reduction="mean")
    
                if engine:                       # DeepSpeed
                    engine.zero_grad(); engine.backward(loss); engine.step()
                else:                            # plain PyTorch
                    adam_optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
                    adam_optimizer.step()
    
                # regularisers / replay
                if args.use_si  and si : si.update_weights(real_model)
                if args.use_ewc and ewc:
                    for ew in ewc: loss += args.lambda_ewc * ew.penalty(real_model)
                if replay_buffer and len(replay_buffer.buffer):
                    loss += replay_buffer.replay_and_calculate_loss(
                        model=real_model, tokenizer=tokenizer,
                        replay_batch_size=args.replay_batch_size,
                        device=device, alpha=args.replay_buffer_weight,
                    )
                epoch_loss += loss.item()
    
            # ——————— epoch summary ———————
            avg_loss = epoch_loss / len(dataloader)
            logging.info(f"[Phase 1] Epoch {epoch}/{PRETRAIN_EPOCHS} — avg loss {avg_loss:.4f}")
    
            # 1️⃣ barrier before the sample
            if dist.is_initialized():
                dist.barrier()
    
            # —— light single-example sample *on every rank* ——
            small_batch = {
                "input_ids": batch["input_ids"][:1].to(device),
                "labels"   : batch["labels"][:1],
            }
            model_for_sample = engine.module if engine else real_model
            sample_results   = quick_eval_sample(
                model_for_sample,
                tokenizer,
                small_batch,
                device=device,
                open_ids=open_ids,
                close_id=close_id,
                m_len=m_len,
            )
    
            # print only on rank-0, but every rank ran the forward → no NCCL mismatch
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                for i, (true_p, pred) in enumerate(sample_results):
                    print(f"[Epoch {epoch} sample {i}] true → {true_p:.2f}, pred → {pred}")
    
            # 2️⃣ barrier after the sample
            if dist.is_initialized():
                dist.barrier()
    
            # ——————— checkpoint ———————
            save_checkpoint(
                engine   = engine if use_deepspeed else None,
                model    = real_model,
                save_dir = args.save_dir,
                tag      = STAGE_TAG[1],
                bucket   = args.bucket,
            )
    
        if getattr(args, "stage_1_only", False):
            logging.info("stage_1_only=True → stopping after Phase 1")
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
    
        # Prepare the token IDs for the opener and closer tags
        open_ids = tokenizer("<STOCK PRICE 30 DAYS OUT>: ", add_special_tokens=False).input_ids
        m_len    = len(open_ids)
        close_id = tokenizer.convert_tokens_to_ids("</STOCK PRICE 30 DAYS OUT>")
    
        fw_count = 0  # running count of forward passes
    
        for epoch in range(1, ebm_epochs + 1):
            real_model.ebm.train()
            total_loss = 0.0
            steps      = 0
    
            for stage in range(3, 4):
                loader = prepare_ft_dataloader(
                    tokenizer,
                    block_size = config.BLOCK_SIZE,
                    shuffle    = True,
                    args       = args,
                    stage      = stage,
                    streaming  = True
                )
    
                for idx, batch in enumerate(loader):
                    print(f"[Phase 4] stage {stage} batch {idx+1}/{len(loader)}")
    
                    # (B, K, T) and (B,)
                    ids        = batch["input_ids"].to(device)
                    true_price = batch["labels"].to(device)
                    B, K, T    = ids.size()
    
                    # 1) Embedding-based forward passes
                    flat_ids = ids.view(B*K, T)
                    with torch.no_grad():
                        embs = real_model.get_embeddings(flat_ids, pool=True)  # (B*K, D)
                    fw_count += B * K
                    print(f"[Phase 4] fw_count {fw_count} (embeddings)")
    
                    embs = embs.view(B, K, -1)  # (B, K, D)
    
                    # 2) Greedy generation per candidate
                    preds = torch.empty(B, K, device=device)
                    for b in range(B):
                        for k in range(K):
                            seq      = ids[b, k]          # (T,)
                            seq_list = seq.tolist()
                            # find opener tag
                            try:
                                j = next(i for i in range(T - m_len + 1)
                                         if seq_list[i:i+m_len] == open_ids)
                            except StopIteration:
                                preds[b,k] = float("inf")
                                continue
    
                            # build prefix including the opener
                            prefix     = seq[: j + m_len].unsqueeze(0)  # (1, L)
                            cur        = prefix.clone()
                            gen_tokens = []
    
                            # generate until close_id
                            for _ in range(10):
                                hs     = real_model.forward_embeddings_only(cur)  # (1, t, D)
                                logits = real_model.lm_head(hs)[:, -1, :]        # (1, V)
                                nxt    = logits.argmax(-1, keepdim=True)         # (1,1)
                                gen_tokens.append(nxt)
                                if nxt.item() == close_id:
                                    break
                                cur = torch.cat([cur, nxt], dim=1)
    
                            # decode & extract value
                            full_ids = torch.cat([prefix, torch.cat(gen_tokens, dim=1)], dim=1)[0]
                            text     = tokenizer.decode(full_ids, skip_special_tokens=False)
                            val      = extract_label_value(text)
                            preds[b,k] = val if val is not None else float("inf")
    
                            fw_count += len(gen_tokens)
                            print(fw_count)
    
                    # 3) Residuals → select positive index
                    true_exp  = true_price.unsqueeze(1).expand_as(preds)
                    residuals = torch.abs(preds - true_exp)  # (B, K)
                    pos_idx   = residuals.argmin(dim=1)      # (B,)
    
                    # 4) EBM scoring
                    flat_embs = embs.view(B*K, -1).to(torch.bfloat16)
                    energies  = real_model.ebm(flat_embs).view(B, K)
                    fw_count += B * K
                    print(fw_count)
    
                    # 5) Margin-ranking loss
                    pos_en = energies[torch.arange(B), pos_idx]
                    mask   = torch.ones_like(energies, dtype=torch.bool)
                    mask[torch.arange(B), pos_idx] = False
                    neg_en = energies[mask].view(B, K-1)
                    loss   = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
                    print(loss, "loss!!")
    
                    ebm_optimizer.zero_grad()
                    loss.backward()
                    ebm_optimizer.step()
    
                    total_loss += loss.item()
                    steps      += 1
    
            logging.info(f"[EBM FT] Epoch {epoch}/{ebm_epochs} — Avg Loss {total_loss/steps:.4f}")
    
        # ---------------------------------------------------------------- #
        #  Validation (stage 8)                                            #
        # ---------------------------------------------------------------- #
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
            for idx, batch in enumerate(val_loader):
                print(f"[Phase 4 Val] batch {idx+1}/{len(val_loader)}")
    
                ids        = batch["input_ids"].to(device)
                true_price = batch["labels"].to(device)
                B, K, T    = ids.size()
    
                # embeddings
                flat_ids = ids.view(B*K, T)
                embs     = real_model.get_embeddings(flat_ids, pool=True).view(B, K, -1)
                fw_count += B * K
                print(fw_count, "fw_count1")
    
                # generation + preds
                preds = torch.empty(B, K, device=device)
                for b in range(B):
                    for k in range(K):
                        seq      = ids[b, k]
                        seq_list = seq.tolist()
                        try:
                            j = next(i for i in range(T - m_len + 1)
                                     if seq_list[i:i+m_len] == open_ids)
                        except StopIteration:
                            preds[b,k] = float("inf")
                            continue
    
                        prefix     = seq[: j + m_len].unsqueeze(0)
                        cur        = prefix.clone()
                        gen_tokens = []
                        for _ in range(10):
                            hs     = real_model.forward_embeddings_only(cur)
                            logits = real_model.lm_head(hs)[:, -1, :]
                            nxt    = logits.argmax(-1, keepdim=True)
                            gen_tokens.append(nxt)
                            if nxt.item() == close_id:
                                break
                            cur = torch.cat([cur, nxt], dim=1)
    
                        full_ids = torch.cat([prefix, torch.cat(gen_tokens, dim=1)], dim=1)[0]
                        text     = tokenizer.decode(full_ids, skip_special_tokens=False)
                        val      = extract_label_value(text)
                        preds[b,k] = val if val is not None else float("inf")
    
                        fw_count += len(gen_tokens)
                        print(fw_count, "fw_count2")
    
                # residuals & pos_idx
                true_exp  = true_price.unsqueeze(1).expand_as(preds)
                residuals = torch.abs(preds - true_exp)
                pos_idx   = residuals.argmin(dim=1)
    
                # EBM scoring + loss
                energies = real_model.ebm(embs.view(B*K, -1).to(torch.bfloat16)).view(B, K)
                fw_count += B * K
                print(fw_count, "fw_count3")
    
                pos_en = energies[torch.arange(B), pos_idx]
                mask   = torch.ones_like(energies, dtype=torch.bool)
                mask[torch.arange(B), pos_idx] = False
                neg_en = energies[mask].view(B, K-1)
                loss   = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
                print(loss, "loss!!")
    
                val_loss += loss.item()
                steps    += 1
    
        logging.info(f"[EBM Val] Avg Loss {val_loss/steps:.4f}")
    
        # ---------------------------------------------------------------- #
        #  Save final model + EBM                                         #
        # ---------------------------------------------------------------- #
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
