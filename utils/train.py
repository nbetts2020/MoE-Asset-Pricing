import os
import gc
import time
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import logging
import subprocess
import re
import itertools
from itertools import chain

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
    0: "common_crawl_base",
    1: "normal_pretrained",
    2: "continual_pretrained_64k",
    3: "continual_pretrained_64k", #"supervised_finetuned_coconut",
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

def extract_label_value(decoded_text: str):
    """
    Return the float that appears after
      '<STOCK PRICE 30 DAYS OUT>:'
    and before the closing tag, ignoring any whitespace or
    stray characters (e.g. '≈', new-lines) that come
    after the number.

    Examples it accepts
    ------------------------------------------
    <STOCK PRICE 30 DAYS OUT>: 28.05</STOCK ...
    <STOCK PRICE 30 DAYS OUT>: 28.05  ≈</STOCK ...
    <STOCK PRICE 30 DAYS OUT>:
        28.05
    </STOCK ...
    """
    pattern = (r'<STOCK PRICE 30 DAYS OUT>:'      # opening tag
               r'\s*'                             # optional spaces/new-lines
               r'([\d\.]+)\b'                     # the number (group 1)
               r'[\s\S]*?'                        # anything, lazily
               r'</STOCK PRICE 30 DAYS OUT>')     # closing tag

    m = re.search(pattern, decoded_text, flags=re.DOTALL)
    if not m:
        return None

    # collapse any accidental '..' into a single '.'
    num_str = re.sub(r'\.\.+', '.', m.group(1))
    try:
        return float(num_str)
    except ValueError:
        logging.error(f"Bad float '{num_str}' in: {decoded_text!r}")
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
    if min_stage >= 0:
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
    #  PHASE 0 – Common Crawl Base Pre-Train                                #
    # ===================================================================== #
    if 0 in args.stages:
        dbg("→ entering Phase 0")

        if args.bucket:
            download_models_from_s3(bucket=args.bucket)
            dist.barrier()

        # ------------------------------------------------------------------ #
        #  0. restore the Phase-1 weights (module-only)                       #
        # ------------------------------------------------------------------ #
        dbg("loading Phase-1 checkpoint")
        # engine.load_checkpoint(args.save_dir, tag=STAGE_TAG[0])
        engine.load_checkpoint(
            load_dir                 = args.save_dir,
            tag                      = STAGE_TAG[0],
            load_optimizer_states    = False,
            load_lr_scheduler_states = False,
            load_module_only         = True,
        )
        real_model = engine.module

        # curriculum: {block_size : epochs}
        CONTEXT_CURRICULUM = {
            4096: 1
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

            # ── 5. train for n_ep epochs at this length ────────────────────
            logging.info(f"[Phase 0] block_size {blk_sz:,} for {n_ep} epoch(s)")
            for ep in range(1, n_ep + 1):
                real_model.train()
                ep_loss = 0.0

                loader = prepare_ft_dataloader(
                    tokenizer,
                    block_size=blk_sz,
                    shuffle=False,
                    args=args,
                    stage=0,         # one call to get 2016–2024 merged
                    streaming=False
                )

                for step, batch in enumerate(loader):
                    print(step, "progress!!")
                    input_ids = pad_to_global(batch["input_ids"].to(device))
                    loss      = real_model.forward_next_token_efficient(input_ids)
                    print(loss, blk_sz, "loss!!")

                    engine.zero_grad()
                    engine.backward(loss)
                    engine.step()

                    ep_loss += loss.item()

                    if step % 1500 == 0:
                        # ── 6. optional checkpoint at this length ──────────────────────
                        save_checkpoint(
                            engine   = engine if use_deepspeed else None,
                            model    = real_model,
                            save_dir = args.save_dir,
                            tag      = f"{STAGE_TAG[0]}",
                            bucket   = args.bucket,
                        )

                avg = ep_loss / (step + 1)
                logging.info(f"[Phase 0] bs={blk_sz:,}  Epoch {ep}/{n_ep}  avg loss {avg:.4f}")

                # ── 6. optional checkpoint at this length ──────────────────────
                save_checkpoint(
                    engine   = engine if use_deepspeed else None,
                    model    = real_model,
                    save_dir = args.save_dir,
                    tag      = f"{STAGE_TAG[0]}",
                    bucket   = args.bucket,
                )

            # stop early if user asked for only stage 2 -------------------------
            if getattr(args, "stage_0_only", False):
                logging.info("stage_0_only=True → stopping after Phase 2")
                return
    
    # ------------------------------------------------------------------ #
    #  PHASE 1  –  curriculum pre-training (single shared dataloader)
    # ------------------------------------------------------------------ #
    if 1 in args.stages:
        dbg("→ entering Phase 1")
        if args.bucket:
            download_models_from_s3(bucket=args.bucket)
            dist.barrier()

        # ------------------------------------------------------------------ #
        #  0. restore the Phase-1 weights (module-only)                       #
        # ------------------------------------------------------------------ #
        dbg("loading Phase-1 checkpoint")
        # engine.load_checkpoint(args.save_dir, tag=STAGE_TAG[0])
        engine.load_checkpoint(
            load_dir                 = args.save_dir,
            tag                      = STAGE_TAG[0],
            load_optimizer_states    = False,
            load_lr_scheduler_states = False,
            load_module_only         = True,
        )
        real_model = engine.module
        dbg("loaded Phase-0 model")

        # map {block_size: num_epochs}
        CURRICULUM = {
            4096 : 1,
            8192: 1,
            16384: 1
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

            engine.optimizer.zero_grad(set_to_none=True)

            logging.info(f"[Phase 1] block_size {blk_sz:,} for {n_ep} epoch(s)")

            for ep in range(1, n_ep + 1):
                real_model.train()
                epoch_loss = 0.0

                # ———————————— mini-batch loop ————————————
                for step, batch in enumerate(dataloader):
                    print(step, "progress!!")
                    input_ids = pad_to_global(batch["input_ids"].to(device))   # pads/trims to blk_sz
                    loss      = real_model.forward_next_token_efficient(input_ids)
                    print(loss, "loss!!")
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
                    if step % 1000 == 0:
                        save_checkpoint(
                            engine   = engine if use_deepspeed else None,
                            model    = real_model,
                            save_dir = args.save_dir,
                            tag      = f"{STAGE_TAG[1]}",
                            bucket   = args.bucket,
                        )

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
            16384 : 1,
            32768  : 1,
            #65536 : 1
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

            # ── 5. train for n_ep epochs at this length ────────────────────
            logging.info(f"[Phase 2] block_size {blk_sz:,} for {n_ep} epoch(s)")
            for ep in range(1, n_ep + 1):
                real_model.train()
                ep_loss = 0.0

                stages = [1,2,3,4,5,6]
                loaders = [
                    prepare_ft_dataloader(tokenizer, block_size=blk_sz, shuffle=False, args=args, stage=s, streaming=False) for s in stages
                ]
                continual_loader = chain(*loaders)

                for step, batch in enumerate(continual_loader):
                    print(step, "progress!!")
                    input_ids = pad_to_global(batch["input_ids"].to(device))
                    loss      = real_model.forward_next_token_efficient(input_ids)
                    print(loss, blk_sz, "loss!!")

                    engine.zero_grad()
                    engine.backward(loss)
                    engine.step()

                    ep_loss += loss.item()
                    if step % 500 == 0:
                        save_checkpoint(
                            engine   = engine if use_deepspeed else None,
                            model    = real_model,
                            save_dir = args.save_dir,
                            tag      = f"{STAGE_TAG[2]}",
                            bucket   = args.bucket,
                        )

                avg = ep_loss / (step + 1)
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
                tag      = STAGE_TAG[3],
                bucket   = args.bucket,
            )

    # ------------------------------------------------------------------ #
    #  PHASE 4 – EBM fine-tuning (+ validation)                          #
    #           marker-pool + L2-norm + LayerNorm head                   #
    #           hard-negative hinge, temp/top-p generation, acc metric   #
    # ------------------------------------------------------------------ #
    if 4 in args.stages:
        logging.info("=== Starting EBM Fine-Tuning Phase ===")

        # -------------------------------------------------------------- #
        #  Restore Phase-2 weights (merge ZeRO-2 → single fp32)          #
        # -------------------------------------------------------------- #
        ckpt_dir   = os.path.join(args.save_dir, STAGE_TAG[2])
        merged_dir = merge_zero2_shards(args.save_dir, tag=STAGE_TAG[2])
        fp32_path  = os.path.join(merged_dir, "mp_rank_00_model_states.pt")
        logging.info(f"[load] torch.load → {fp32_path}")
        fp32_state = torch.load(fp32_path, map_location="cpu")["module"]
        torch.save(fp32_state, f"{merged_dir}_fp32.pth")          # cache

        real_model = engine.module if use_deepspeed else model
        real_model.load_state_dict(fp32_state, strict=False)

        # -------------------------------------------------------------- #
        #  Freeze backbone – train only the EBM head                     #
        # -------------------------------------------------------------- #
        real_model.eval()
        for p in real_model.parameters():
            p.requires_grad_(False)

        # add a LayerNorm in front of the head (place it on the same GPU)
        if not isinstance(real_model.ebm.fc[0], nn.LayerNorm):
            ln = nn.LayerNorm(real_model.ebm.fc[0].in_features, dtype=torch.bfloat16).to(device)
            real_model.ebm.fc = nn.Sequential(ln, *list(real_model.ebm.fc))

        for p in real_model.ebm.parameters():
            p.requires_grad_(True)

        ebm_opt    = torch.optim.AdamW(real_model.ebm.parameters(), lr=1e-5)
        margin     = 0.0
        ebm_epochs = getattr(args, "ebm_epochs", 1)

        # -------------------------------------------------------------- #
        #  Marker-pool helper                                            #
        # -------------------------------------------------------------- #
        POOL_STRATEGY = "last_token"      # "last_token" | "single_head" | "multi_head"
        POOL_WINDOW   = -1                # –1 ⇒ all tokens before marker
        NUM_HEADS     = 4                  # used only for "multi_head"
        # ──────────────────────────────────────────────────────────────────────────────

        D = real_model.n_embed

        # learnable query/queries for attention strategies
        if POOL_STRATEGY == "single_head":
            pool_q = nn.Parameter(torch.randn(D, device=device))
            real_model.register_parameter("pool_q", pool_q)
        elif POOL_STRATEGY == "multi_head":
            if D % NUM_HEADS:
                raise ValueError(f"D={D} not divisible by NUM_HEADS={NUM_HEADS}")
            head_dim = D // NUM_HEADS
            pool_q = nn.Parameter(torch.randn(NUM_HEADS, head_dim, device=device))
            real_model.register_parameter("pool_q", pool_q)

        # constant IDs we need every call
        open_ids = tokenizer("<STOCK PRICE 30 DAYS OUT>: ",
                             add_special_tokens=False).input_ids
        m_len    = len(open_ids)
        open_id = tokenizer("<STOCK PRICE 30 DAYS OUT>: ",
                            add_special_tokens=False).input_ids[0]
        delim_ids    = tokenizer("Last 8 Articles for the Current Stock",
                                 add_special_tokens=False).input_ids
        delim_len    = len(delim_ids)
        delim_tensor = torch.tensor(delim_ids, device=device)
        close_id = tokenizer.convert_tokens_to_ids("</STOCK PRICE 30 DAYS OUT>")

        pre_ln = {}

        def tap_pre_ln(module, input, output):
            # input[0] is the tensor before the final LayerNorm
            pre_ln['x'] = input[0].detach()
            return output  # keep normal forward flow

        # Register this once, after building/loading your model
        handle = real_model.ln_f.register_forward_hook(tap_pre_ln)

        def marker_pool(ids_batch: torch.Tensor) -> torch.Tensor:
            """
            1) For each row, truncate at the delimiter “Last 8 Articles for the Current
               Stock”, append the <STOCK PRICE 30 DAYS OUT>: token.
            2) Pad/trim to block_size and run the model to get hidden states (pre-LN).
            3) Pool according to POOL_STRATEGY.
            Returns (B, D).
            """
            B, _ = ids_batch.shape
            device = ids_batch.device
            seqs = []

            # build per-row sequences (prefix + marker)
            for b in range(B):
                row = ids_batch[b]
                split_idx = next(
                    (i for i in range(row.size(0) - delim_len + 1)
                     if torch.equal(row[i:i+delim_len], delim_tensor)),
                    row.size(0)
                )
                prefix = row[:split_idx]
                seqs.append(torch.cat([prefix, torch.tensor([open_id], device=device)]))

            # pad/trim each to model.block_size and stack
            padded = torch.stack([
                real_model._pad_or_trim(s.unsqueeze(0), real_model.block_size).squeeze(0)
                for s in seqs
            ], dim=0)  # (B, block_size)

            # run full forward to trigger our pre-LN hook (we ignore the output)
            _ = real_model(padded)

            # retrieve the pre-LN hidden states
            hs_raw = pre_ln['x']  # (B, block_size, D)
            marker_idx = hs_raw.size(1) - 1

            # pooling
            if POOL_STRATEGY == "last_token":
                return hs_raw[:, marker_idx, :]  # (B, D)

            # window of tokens before the marker
            if POOL_WINDOW == -1:
                start = 0
            else:
                start = max(0, marker_idx - POOL_WINDOW)
            window = hs_raw[:, start:marker_idx, :]  # (B, L, D)

            if POOL_STRATEGY == "single_head":
                scores = torch.matmul(window, pool_q) / math.sqrt(D)  # (B, L)
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)   # (B, L, 1)
                return (weights * window).sum(dim=1)                   # (B, D)

            else:  # multi_head
                H, d_h = pool_q.shape  # (H, head_dim)
                window = window.view(B, window.size(1), H, d_h)  # (B, L, H, d_h)
                scores = (window * pool_q).sum(-1) / math.sqrt(d_h)  # (B, L, H)
                weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, H, 1)
                pooled = (weights * window).sum(1)                    # (B, H, d_h)
                return pooled.view(B, D)                              # (B, D)

        # -------------------------------------------------------------- #
        #  TRAIN EBM                                                     #
        # -------------------------------------------------------------- #
        for epoch in range(1, ebm_epochs + 1):
            real_model.ebm.train()
            total_loss, steps, correct = 0.0, 0, 0

            loader = prepare_ft_dataloader(tokenizer,
                                           block_size=config.BLOCK_SIZE,
                                           shuffle=True,
                                           args=args, stage=7,
                                           streaming=False)
            for step, batch in enumerate(loader):
                if step < 5:
                    continue
                if step == 9:
                    break
                ids, true_price = batch["input_ids"].to(device), batch["labels"].to(device)
                B, K, T = ids.shape
                preds   = torch.zeros((B, K), device=device)
                # ---- 1. marker-pooled, L2-normalised embeddings -------------
                embs = torch.stack([marker_pool(ids[:, k, :]) for k in range(K)], dim=1)  # (B,K,D)

                # batch‐wide z-score
                mean = embs.mean(dim=(0,1), keepdim=True)                                 # (1,1,D)
                std  = embs.std(dim=(0,1), keepdim=True).clamp(min=1e-5)                  # (1,1,D)
                embs = (embs - mean) / std                                                # (B,K,D)

                # **compute diagnostics** before printing**
                var  = embs.var(dim=(0,1), unbiased=False).mean().item()                  # scalar ≈1
                norm = embs.norm(dim=-1).mean().item()                                    # avg L2 norm

                print(f"Avg var: {var:.4e}, Avg norm: {norm:.2f}, SNR: {var/(norm**2):.2%}")
                print(f"[DEBUG] Avg embedding-var across K: {var:.6f}")
                # ---- 2. temperature / top-p sampling to predict price ------- #
                temperature, top_p = 0.8, 0.9
                for b in range(B):
                    for k in range(K):
                        seq = ids[b, k].tolist()
                        try:
                            j = next(i for i in range(T - m_len + 1)
                                     if seq[i:i+m_len] == open_ids)
                        except StopIteration:
                            continue
                        prefix    = ids[b, k, : j+m_len].unsqueeze(0)           # (1,L)
                        generated = prefix.clone()
                        with torch.no_grad():
                            for _ in range(10):
                                full_hs  = real_model.forward_embeddings_only(generated)
                                hs_slice = full_hs[:, : generated.size(1), :]
                                logits   = real_model.lm_head(hs_slice)[:, -1, :]
                                logits  /= temperature

                                sorted_logits, sorted_idx = logits.sort(descending=True)
                                cprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                                remove = cprobs > top_p
                                remove[..., 1:] = remove[..., :-1].clone()
                                remove[..., 0]  = False
                                logits[0, sorted_idx[0, remove[0]]] = float("-inf")

                                probs = torch.softmax(logits, dim=-1)
                                nxt   = torch.multinomial(probs, 1)
                                generated = torch.cat([generated, nxt], dim=1)
                                if nxt.item() == close_id:
                                    break
                        # ensure close-tag
                        if generated[0, -1].item() != close_id:
                            generated = torch.cat([generated,
                                torch.tensor([[close_id]], device=device)], dim=1)

                        txt = tokenizer.decode(generated[0], skip_special_tokens=False)
                        val = extract_label_value(txt)
                        print(txt, step)
                        if val is not None:
                            preds[b, k] = val

                # ---- 3. hard-negative hinge loss --------------------------- #
                energies = real_model.ebm(
                              embs.view(B*K, -1).to(torch.bfloat16)
                           ).view(B, K)
                print(energies)
                pos_idx = (preds - true_price[:, None]).abs().argmin(dim=1)
                pos_en  = energies[torch.arange(B), pos_idx]

                # mask True for the positive example only
                mask = torch.zeros_like(energies, dtype=torch.bool)
                mask[torch.arange(B), pos_idx] = True

                # set positives to +∞ so only negatives remain for the min
                neg_en = energies.masked_fill(mask, float("inf")).min(dim=1).values

                loss = F.relu(margin + pos_en - neg_en).mean()

                correct += ((pos_en + margin) < neg_en).float().sum().item()
                ebm_opt.zero_grad(); loss.backward(); ebm_opt.step()
                total_loss += loss.item(); steps += 1

            train_acc = correct / (steps * B)
            cls_acc = (energies.argmin(dim=1) == pos_idx).float().mean()
            print(cls_acc, "cls_acc")
            logging.info(f"[EBM FT] Epoch {epoch}/{ebm_epochs}  "
                         f"AvgLoss={total_loss/steps:.4f}  SepAcc={train_acc:.3f}")

        # -------------------------------------------------------------- #
        #  VALIDATION                                                    #
        # -------------------------------------------------------------- #
        logging.info("=== EBM Validation (stage 8) ===")
        real_model.ebm.eval()
        val_loader = prepare_ft_dataloader(tokenizer,
                                           block_size=config.BLOCK_SIZE,
                                           shuffle=False,
                                           args=args, stage=8,
                                           streaming=True)

        val_loss, steps, correct = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids, true_price = batch["input_ids"].to(device), batch["labels"].to(device)
                B, K, _ = ids.shape

                embs = torch.stack([marker_pool(ids[:, k, :]) for k in range(K)], dim=1)
                embs = F.normalize(embs, dim=-1)
                energies = real_model.ebm(
                              embs.view(B*K, -1).to(torch.bfloat16)
                           ).view(B, K)

                preds = torch.zeros((B, K), device=device)
                for b in range(B):
                    for k in range(K):
                        seq_ids = ids[b, k]
                        if close_id not in seq_ids:
                            seq_ids = torch.cat([seq_ids,
                                       torch.tensor([close_id], device=device)])
                        preds[b, k] = extract_label_value(
                            tokenizer.decode(seq_ids, skip_special_tokens=False)
                        ) or 0.0

                pos_idx = (preds - true_price[:, None]).abs().argmin(dim=1)
                pos_en  = energies[torch.arange(B), pos_idx]


                # mask True for the positive example only
                mask = torch.zeros_like(energies, dtype=torch.bool)
                mask[torch.arange(B), pos_idx] = True

                # set positives to +∞ so only negatives remain for the min
                neg_en = energies.masked_fill(mask, float("inf")).min(dim=1).values

                loss = F.relu(margin + pos_en - neg_en).mean()

                correct += ((pos_en + margin) < neg_en).float().sum().item()
                val_loss += loss.item(); steps += 1

        val_acc = correct / (steps * B)
        logging.info(f"[EBM Val] AvgLoss={val_loss/steps:.4f}  SepAcc={val_acc:.3f}")

        # -------------------------------------------------------------- #
        #  SAVE                                                          #
        # -------------------------------------------------------------- #
        save_checkpoint(
            engine   = engine if use_deepspeed else None,
            model    = real_model,
            save_dir = args.save_dir,
            tag      = STAGE_TAG[4],
            bucket   = args.bucket,
        )

        logging.info("=== Phase 4 complete ===")

    logging.info("Training complete.")
