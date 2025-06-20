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

from utils.utils import compute_l2_loss, upload_checkpoint_to_s3, download_models_from_s3
from utils.data import prepare_ft_dataloader
from utils.config import config
from utils.ewc import ElasticWeightConsolidation
from utils.si import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
import deepspeed

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# Import the helper function to update RoPE buffers.
from utils.model import update_model_rope_for_extended_context

STAGE_TAG = {
    1: "normal_pretrained",
    2: "continual_pretrained_64k",
    3: "supervised_finetuned_coconut",
    4: "model_with_ebm",
}

# ------------------------------------------------------------
# helper  ➜  merge all ZeRO-2 mp_rank_XX_model_states.pt files
# ------------------------------------------------------------
import glob, shutil

def merge_zero2_shards(ckpt_root: str, tag: str = "continual_pretrained_64k") -> str:
    """
    ckpt_root/
        continual_pretrained_64k/
            mp_rank_00_model_states.pt
            mp_rank_01_model_states.pt
            ...
    ───────────────────────────────────────────────────────────
    Collects every `mp_rank_??_model_states.pt`, merges the
    'module' dicts, and writes a *single* shard into

        ckpt_root/<tag>_merged/mp_rank_00_model_states.pt

    Returns that new directory so you can hand it to DeepSpeed.
    """
    src_dir   = os.path.join(ckpt_root, tag)
    shard_paths = sorted(glob.glob(os.path.join(src_dir, "mp_rank_*_model_states.pt")))
    assert shard_paths, f"No ZeRO shards found in {src_dir}"

    merged = None
    for p in shard_paths:
        shard = torch.load(p, map_location="cpu")
        if merged is None:
            merged = shard
        else:
            merged["module"].update(shard["module"])    # <- add missing keys

    tgt_dir = os.path.join(ckpt_root, f"{tag}_merged")
    os.makedirs(tgt_dir, exist_ok=True)
    torch.save(merged, os.path.join(tgt_dir, "mp_rank_00_model_states.pt"))

    # (optional) copy the tokenizer / config json if you stored them there
    for fname in ("latest", "ds_inference_config.json"):
        fsrc = os.path.join(src_dir, fname)
        if os.path.isfile(fsrc):
            shutil.copy2(fsrc, tgt_dir)

    logging.info(f"[merge] wrote merged ZeRO-2 checkpoint to {tgt_dir}")
    return tgt_dir

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

    After writing, if `bucket` is set, all DeepSpeed ranks will
    synchronize and then each upload its own shard to S3. In the
    pure‐PyTorch path (engine=None), only rank 0 uploads.
    """
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

    if engine is not None:
        # ── DeepSpeed path: every rank writes its own shard locally ──
        if rank0:
            print(f"[save] DeepSpeed checkpoint → {save_dir} (tag={tag})")
        engine.save_checkpoint(save_dir, tag=tag)

        if bucket:
            # wait for all ranks to finish writing
            if dist.is_initialized():
                dist.barrier()

            # now each rank uploads whatever is in save_dir/<tag>/ to S3
            upload_checkpoint_to_s3(
                local_dir=os.path.join(save_dir, tag),
                bucket=bucket,
                remote_dir=f"model/{tag}"
            )

    else:
        # ── plain PyTorch path: only rank 0 saves and uploads ──
        ckpt_dir = os.path.join(save_dir, tag)
        if rank0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pth"))

            if bucket:
                upload_checkpoint_to_s3(
                    local_dir=ckpt_dir,
                    bucket=bucket,
                    remote_dir=f"model/{tag}"
                )

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
    # ------------------------------------------------------------------ #
    #  PHASE 1  –  curriculum pre-training (single shared dataloader)
    # ------------------------------------------------------------------ #
    if 1 in args.stages:
        # map {block_size: num_epochs}
        CURRICULUM = {
            128 : 10,
            512 : 2,
            2048: 1,
            4096: 1,
        }

        total_epochs = sum(CURRICULUM.values())
        logging.info(f"→ Phase 1: curriculum pre-training ({total_epochs} total epochs)")

        open_ids = tokenizer("<STOCK PRICE 30 DAYS OUT>: ", add_special_tokens=False).input_ids
        m_len    = len(open_ids)
        close_id = tokenizer.convert_tokens_to_ids("</STOCK PRICE 30 DAYS OUT>")

        for blk_sz, n_ep in CURRICULUM.items():

            # ── bump context length & RoPE tables ──────────────────────────
            real_model.block_size = blk_sz
            config.BLOCK_SIZE     = blk_sz
            update_model_rope_for_extended_context(real_model, blk_sz)

            if dist.is_initialized() and blk_sz % dist.get_world_size() != 0:
                raise ValueError(f"block_size {blk_sz} must be divisible by world_size")

            logging.info(f"[Phase 1] block_size {blk_sz:,} for {n_ep} epoch(s)")

            for ep in range(1, n_ep + 1):
                real_model.train()
                epoch_loss = 0.0

                # ———————————— mini-batch loop ————————————
                for step, batch in enumerate(dataloader):
                    input_ids = pad_to_global(batch["input_ids"].to(device))   # pads/trims to blk_sz
                    loss      = real_model.forward_next_token_efficient(input_ids)

                    if engine:            # DeepSpeed
                        engine.zero_grad(); engine.backward(loss); engine.step()
                    else:                 # plain PyTorch
                        adam_optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
                        adam_optimizer.step()

                    # regularisers / replay (unchanged) ---------------------
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

                avg_loss = epoch_loss / max(1, len(dataloader))
                logging.info(f"[Phase 1] bs={blk_sz:,}  Epoch {ep}/{n_ep}  avg loss {avg_loss:.4f}")

                # optional checkpoint per context length ------------------------
                save_checkpoint(
                    engine   = engine if use_deepspeed else None,
                    model    = real_model,
                    save_dir = args.save_dir,
                    tag      = f"{STAGE_TAG[1]}",
                    bucket   = args.bucket,
                )

            if getattr(args, "stage_1_only", False):
                logging.info("stage_1_only=True → stopping after Phase 1")
                return

    # ===================================================================== #
    #  PHASE 2 – continual pre-training with progressively longer contexts  #
    # ===================================================================== #
    if 2 in args.stages:
        dbg("→ entering Phase 2")
        if args.bucket:
            download_models_from_s3(bucket=args.bucket)
            dist.barrier()

        # ------------------------------------------------------------------ #
        #  0. restore the Phase-1 weights (module-only)                       #
        # ------------------------------------------------------------------ #
        dbg("loading Phase-1 checkpoint")
        # engine.load_checkpoint(args.save_dir, tag=STAGE_TAG[1])
        engine.load_checkpoint(
            load_dir                 = args.save_dir,
            tag                      = STAGE_TAG[1],
            load_optimizer_states    = False,
            load_lr_scheduler_states = False,
            load_module_only         = True,
        )
        real_model = engine.module
        dbg("loaded Phase-1 model")

        # curriculum: {block_size : epochs}
        CONTEXT_CURRICULUM = {
            4096: 2,
            8192  : 1,
            16384 : 1
        }

        for blk_sz, n_ep in CONTEXT_CURRICULUM.items():
            # ── 1. sync to be safe ──────────────────────────────────────────
            if dist.is_initialized(): dist.barrier()

            if config.BLOCK_SIZE != blk_sz:
                # ── 2. resize context + RoPE ───────────────────────────────────
                real_model.block_size = blk_sz
                config.BLOCK_SIZE     = blk_sz
                update_model_rope_for_extended_context(real_model, blk_sz)

            if blk_sz % (dist.get_world_size() if dist.is_initialized() else 1) != 0:
                raise ValueError(f"block_size {blk_sz} not divisible by world_size")

            # ── 3. clear optimiser state so no old-context momentum leaks ──
            engine.optimizer.zero_grad(set_to_none=True)

            # ── 4. build streaming dataloader for the *new* length ─────────
            dbg(f"building continual_loader (bs={blk_sz:,})")
            # continual_loader = prepare_ft_dataloader(
            #     tokenizer,
            #     block_size = blk_sz,
            #     shuffle    = False,
            #     args       = args,
            #     stage      = 1,          # same dataset as before
            #     streaming  = True,
            # )
            import itertools

            # # build each stage’s loader
            # loader_stage1 = prepare_ft_dataloader(
            #     tokenizer,
            #     block_size = blk_sz,
            #     shuffle    = False,
            #     args       = args,
            #     stage      = 1,
            #     streaming  = True,
            # )
            # loader_stage2 = prepare_ft_dataloader(
            #     tokenizer,
            #     block_size = blk_sz,
            #     shuffle    = False,
            #     args       = args,
            #     stage      = 2,
            #     streaming  = True,
            # )

            # chain them into one
            from itertools import chain
            # continual_loader = chain(loader_stage1, loader_stage2)
            dbg("continual_loader ready")

            # ── 5. train for n_ep epochs at this length ────────────────────
            logging.info(f"[Phase 2] block_size {blk_sz:,} for {n_ep} epoch(s)")
            for ep in range(1, n_ep + 1):
                real_model.train()
                ep_loss = 0.0

                loader_stage1 = prepare_ft_dataloader(
                    tokenizer, block_size=blk_sz, shuffle=False,
                    args=args, stage=1, streaming=True
                )
                loader_stage2 = prepare_ft_dataloader(
                    tokenizer, block_size=blk_sz, shuffle=False,
                    args=args, stage=2, streaming=True
                )
                continual_loader = chain(loader_stage1, loader_stage2)

                for step, batch in enumerate(continual_loader):
                    print(step, "progress!!")
                    input_ids = pad_to_global(batch["input_ids"].to(device))
                    loss      = real_model.forward_next_token_efficient(input_ids)
                    print(loss, blk_sz, "loss!!")

                    engine.zero_grad()
                    engine.backward(loss)
                    engine.step()

                    ep_loss += loss.item()

                avg = ep_loss / 627
                logging.info(f"[Phase 2] bs={blk_sz:,}  Epoch {ep}/{n_ep}  avg loss {avg:.4f}")

                # ── 6. optional checkpoint at this length ──────────────────────
                save_checkpoint(
                    engine   = engine if use_deepspeed else None,
                    model    = real_model,
                    save_dir = args.save_dir,
                    tag      = f"{STAGE_TAG[2]}",
                    bucket   = args.bucket,
                )

            # stop early if user asked for only stage 2 -------------------------
            if getattr(args, "stage_2_only", False):
                logging.info("stage_2_only=True → stopping after Phase 2")
                return

    # --------------------------------------------------------------------- #
    #  PHASE 3 – Reasoning                                                  #
    # --------------------------------------------------------------------- #
    if 3 in args.stages:
        dbg("→ entering Phase 3")
        if args.bucket:
            download_models_from_s3(bucket=args.bucket)
            dist.barrier()
        # ── 0. load the Phase-1 checkpoint before touching RoPE or SP groups ─────
        dbg("loading Phase-2 checkpoint")
        engine.load_checkpoint(args.save_dir, tag=STAGE_TAG[2])
        real_model = engine.module  # unwrap DeepSpeed
        dbg("loaded Phase-2 model")

        device   = torch.device("cuda")

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

        PHASE3_EPOCHS = 100
        for epoch in range(1, PHASE3_EPOCHS + 1):
            print(epoch, PHASE3_EPOCHS)
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

            save_checkpoint(
                engine   = engine if use_deepspeed else None,
                model    = real_model,
                save_dir = args.save_dir,
                tag      = STAGE_TAG[1],
                bucket   = args.bucket,
            )

    # ------------------------------------------------------------------ #
    #  PHASE 4 – EBM fine‑tuning (+ validation) with **marker pooling**   #
    # ------------------------------------------------------------------ #
    if 4 in args.stages:
        logging.info("=== Starting EBM Fine‑Tuning Phase ===")
    
        if args.bucket:
            download_models_from_s3(bucket=args.bucket)
            dist.barrier()
    
        # ------------ assemble one fp32 state_dict -----------------
        ckpt_dir   = os.path.join(args.save_dir, STAGE_TAG[2])      # "continual_pretrained_64k"
        merged_dir = merge_zero2_shards(args.save_dir, tag=STAGE_TAG[2])
        fp32_path  = os.path.join(merged_dir, "mp_rank_00_model_states.pt")
    
        logging.info(f"[load] torch.load → {fp32_path}")
        fp32_state = torch.load(fp32_path, map_location="cpu")["module"]
    
        # cache single‑file version (optional)
        torch.save(fp32_state, f"{merged_dir}_fp32.pth")
    
        # push weights into live model ------------------------------------------------
        real_model = engine.module if use_deepspeed else model
        real_model.load_state_dict(fp32_state, strict=False)
    
        # freeze backbone – train only EBM --------------------------------------------
        real_model.eval()
        for p in real_model.parameters():         p.requires_grad_(False)
        for p in real_model.ebm.parameters():     p.requires_grad_(True)
    
        ebm_opt    = torch.optim.AdamW(real_model.ebm.parameters(), lr=args.ebm_lr)
        margin     = getattr(args, "ebm_margin", 1.0)
        ebm_epochs = getattr(args, "ebm_epochs", 1)
    
        # ─────────── helper: marker‑pooled embedding ────────────
        open_ids   = tokenizer("<STOCK PRICE 30 DAYS OUT>: ", add_special_tokens=False).input_ids
        marker_id  = open_ids[0]                           # first token of the marker
    
        def marker_pool(ids_batch: torch.Tensor) -> torch.Tensor:
            """Return (B,D) embedding pooled at first <STOCK ...> token."""
            # ids_batch: (B,T)
            hs = real_model.forward_embeddings_only(ids_batch)      # (B,T,D)
            B, T = ids_batch.shape
            device = ids_batch.device
            # locate first marker; if none, fall back to global mean
            pos = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                hits = (ids_batch[b] == marker_id).nonzero(as_tuple=True)[0]
                pos[b] = hits[0] if hits.numel() else 0
            return hs[torch.arange(B, device=device), pos, :]        # (B,D)
    
        close_id = tokenizer.convert_tokens_to_ids("</STOCK PRICE 30 DAYS OUT>")
    
        # ----------------------------------------------------------------------
        #  TRAIN EBM
        # ----------------------------------------------------------------------
        for epoch in range(1, ebm_epochs + 1):
            real_model.ebm.train()
            total_loss, steps = 0.0, 0
    
            for stage in range(3, 4):                                              # ft_datasets 3‑7
                loader = prepare_ft_dataloader(tokenizer,
                                               block_size=config.BLOCK_SIZE,
                                               shuffle=True,
                                               args=args, stage=stage,
                                               streaming=True)
                for batch in loader:
                    ids        = batch["input_ids"].to(device)                    # (B,K,T)
                    true_price = batch["labels"].to(device)                       # (B,)
                    B, K, T    = ids.shape
                    preds      = torch.zeros((B, K), device=device)
    
                    # -------- 1. backbone → marker‑pooled embeddings -------------
                    emb_list = []
                    for k_idx in range(K):
                        emb_k = marker_pool(ids[:, k_idx, :])                      # (B,D)
                        emb_list.append(emb_k)
                    embs = torch.stack(emb_list, dim=1)                            # (B,K,D)
    
                    # -------- 2. greedy decode numeric preds (unchanged) ---------
                    temperature, top_p = 0.8, 0.9
                    for b in range(B):
                        for k in range(K):
                            seq = ids[b, k].tolist()
                            # find the “<STOCK PRICE 30 DAYS OUT>: ” marker
                            try:
                                j = next(i for i in range(T - len(open_ids) + 1)
                                         if seq[i:i+len(open_ids)] == open_ids)
                            except StopIteration:
                                continue
    
                            prefix    = ids[b, k, : j + len(open_ids)].unsqueeze(0)
                            generated = prefix.clone()
                            with torch.no_grad():
                                for _ in range(10):
                                    hs = real_model.forward_embeddings_only(generated)    # (1,L,D)
                                    logits = real_model.lm_head(hs)[:, -1, :]
                                    logits /= temperature
                                    sorted_logits, sorted_idx = logits.sort(descending=True)
                                    cprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                                    remove = cprobs > top_p
                                    remove[..., 1:] = remove[..., :-1].clone()
                                    remove[..., 0]  = False
                                    logits[0, sorted_idx[0, remove[0]]] = -float("inf")
                                    probs = torch.softmax(logits, dim=-1)
                                    nxt   = torch.multinomial(probs, 1)
                                    generated = torch.cat([generated, nxt], dim=1)
                                    if nxt.item() == close_id:
                                        break
                            txt = tokenizer.decode(generated[0], skip_special_tokens=False)
                            val = extract_label_value(txt)
                            if val is not None:
                                preds[b, k] = val
    
                    # -------- 3. margin‑ranking loss -----------------------------
                    flat_embs = embs.view(B*K, -1).to(torch.bfloat16)
                    energies  = real_model.ebm(flat_embs).view(B, K)
    
                    pos_idx = (preds - true_price[:, None]).abs().argmin(dim=1)
                    pos_en  = energies[torch.arange(B), pos_idx]
                    neg_mask = torch.ones_like(energies, dtype=torch.bool)
                    neg_mask[torch.arange(B), pos_idx] = False
                    neg_en  = energies[neg_mask].view(B, K-1)
                    loss    = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()
    
                    ebm_opt.zero_grad(); loss.backward(); ebm_opt.step()
                    total_loss += loss.item(); steps += 1
    
            logging.info(f"[EBM FT] Epoch {epoch}/{ebm_epochs}  AvgLoss={total_loss/steps:.4f}")
    
        # ----------------------------------------------------------------------
        #  VALIDATION – same marker pooling + correct positive idx -------------
        # ----------------------------------------------------------------------
        logging.info("=== EBM Validation (stage 8) ===")
        real_model.ebm.eval()
        val_loader = prepare_ft_dataloader(tokenizer,
                                           block_size=config.BLOCK_SIZE,
                                           shuffle=False,
                                           args=args, stage=8,
                                           streaming=True)
        val_loss, steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids        = batch["input_ids"].to(device)                       # (B,K,T)
                true_price = batch["labels"].to(device)
                B, K, T    = ids.shape
    
                emb_list = [marker_pool(ids[:, k, :]) for k in range(K)]          # list[(B,D)]
                embs = torch.stack(emb_list, dim=1)                              # (B,K,D)
                energies = real_model.ebm(embs.view(B*K, -1).to(torch.bfloat16)).view(B, K)
    
                # pick same "closest" candidate as in training
                preds = torch.zeros((B, K), device=device)
                for b in range(B):
                    for k in range(K):
                        preds[b, k] = extract_label_value(
                            tokenizer.decode(ids[b, k], skip_special_tokens=False)
                        ) or 0.0
                pos_idx = (preds - true_price[:, None]).abs().argmin(dim=1)
                pos_en  = energies[torch.arange(B), pos_idx]
                neg_en  = energies.mean(dim=1)
                loss    = F.relu(margin + pos_en - neg_en).mean()
                val_loss += loss.item(); steps += 1
    
        logging.info(f"[EBM Val] AvgLoss={val_loss/steps:.4f}")
    
        # ----------------------------------------------------------------------
        #  SAVE -----------------------------------------------------------------
        # ----------------------------------------------------------------------
        final_tag = "model_with_ebm"
        if use_deepspeed:
            engine.save_checkpoint(args.save_dir, tag=final_tag)
        else:
            torch.save(real_model.state_dict(), os.path.join(args.save_dir, "model_with_ebm.pth"))
    
        save_checkpoint(engine if use_deepspeed else None,
                        real_model, args.save_dir, STAGE_TAG[4], bucket=args.bucket)

        logging.info("=== Phase 4 complete ===")
        
    logging.info("Training complete.")
