import os, re, logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from deepspeed.runtime.zero.stage3 import GatheredParameters

from utils.utils  import compute_l2_loss, upload_checkpoint_to_s3, prepare_optimizer
from utils.data   import prepare_ft_dataloader
from utils.config import config
from utils.ewc    import ElasticWeightConsolidation
from utils.si     import SynapticIntelligence
from utils.memory_replay_buffer import MemoryReplayBuffer
from utils.model  import update_model_rope_for_extended_context, expand_pos_embedding

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _seq_parallel_slice(full_ids, rank, world_size):
    """
    Split the sequence dimension across ranks so each GPU receives a
    contiguous chunk of tokens.  Returns (local_ids, global_offset).
    """
    t_dim  = -1
    T_full = full_ids.size(t_dim)
    assert T_full % world_size == 0, "T_full must be divisible by world_size"
    T_local       = T_full // world_size
    start, end    = rank * T_local, (rank + 1) * T_local
    slicer        = [slice(None)] * full_ids.dim()
    slicer[t_dim] = slice(start, end)
    return full_ids[tuple(slicer)], start


def extract_label_value(decoded_text):
    m = re.search(r'<30 DAY LABEL>:\s*([\d\.]+)', decoded_text)
    if not m:
        return None
    num = re.sub(r'\.\.+', '.', m.group(1))
    try:
        return float(num)
    except ValueError:
        logging.error(f"Cannot convert label: {num}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# training loop
# ────────────────────────────────────────────────────────────────────────────
def train_model(
    model,
    optimizer,
    epochs,
    device,
    dataloader,
    args,
    si:   SynapticIntelligence | None = None,
    ewc:  list[ElasticWeightConsolidation] | None = None,
    replay_buffer: MemoryReplayBuffer | None = None,
    tokenizer=None,
    use_deepspeed: bool = False,
):
    """
    Four-phase training pipeline:

    1. Normal next-token LM pre-training
    2. Continual pre-training with extended 64-K context
    3. Coconut supervised fine-tune
    4. Energy-based model fine-tune / validation
    """

    engine          = model if use_deepspeed else None
    adam_optimizer  = optimizer if not use_deepspeed else None

    rank       = torch.distributed.get_rank()  if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 1 – normal pre-training
    # ───────────────────────────────────────────────────────────────────────
    logging.info(f"[rank {rank}] normal pre-training begins")
    for epoch in range(1, epochs + 1):
        model.train(); epoch_loss = 0.0
        for batch in dataloader:
            full_ids = batch["input_ids"].to(device)
            ids, offset = _seq_parallel_slice(full_ids, rank, world_size)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_next_token_efficient(ids, offset=offset)

            if world_size > 1:
                torch.distributed.all_reduce(loss); loss /= world_size

            if use_deepspeed:
                engine.zero_grad(); engine.backward(loss); engine.step()
            else:
                adam_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                adam_optimizer.step()

            # regularisation & replay
            if args.use_si and si:
                si.update_weights(model)
            if args.use_ewc and ewc:
                for e in ewc:
                    loss += args.lambda_ewc * e.penalty(model)
            if replay_buffer and len(replay_buffer.buffer):
                loss += replay_buffer.replay_and_calculate_loss(
                    model, tokenizer, args.replay_batch_size,
                    device, args.replay_buffer_weight
                )
            epoch_loss += loss.item()

        logging.info(f"[Normal] epoch {epoch}/{epochs} | loss {epoch_loss/len(dataloader):.4f}")

    tag_dir = os.path.join(args.save_dir, "normal_pretrained")
    os.makedirs(tag_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag="normal_pretrained")
    else:
        torch.save(model.state_dict(), os.path.join(tag_dir, "model.pth"))

    if getattr(args, "stage_1_only", False):
        logging.info("stage_1_only flag set – exiting after phase 1.")
        return

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 2 – continual pre-training (64 K context)
    # ───────────────────────────────────────────────────────────────────────
    logging.info("continual pre-training @ 64 K tokens")
    config.BLOCK_SIZE = config.CONTEXT_WINDOW = 65_536
    model.tokenizer.model_max_length = config.CONTEXT_WINDOW
    expand_pos_embedding(model, config.BLOCK_SIZE)
    update_model_rope_for_extended_context(model, config.BLOCK_SIZE)

    adam_optimizer = prepare_optimizer(model, args)["main"]
    if use_deepspeed:
        engine.optimizer = adam_optimizer

    cont_loader = prepare_ft_dataloader(tokenizer, config.BLOCK_SIZE, False, args, stage=1)
    for batch in cont_loader:
        full_ids = batch["input_ids"].to(device)
        ids, offset = _seq_parallel_slice(full_ids, rank, world_size)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                model._gathered_weights = model.token_embedding_table.weight.clone().half()
                loss = model.forward_next_token_efficient(ids, offset=offset)

        if world_size > 1: torch.distributed.all_reduce(loss); loss /= world_size
        if use_deepspeed:
            engine.zero_grad(); engine.backward(loss); engine.step()
        else:
            adam_optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            adam_optimizer.step()

    tag_dir = os.path.join(args.save_dir, "continual_pretrained_64k")
    os.makedirs(tag_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag="continual_pretrained_64k")
    else:
        torch.save(model.state_dict(), os.path.join(tag_dir, "model.pth"))

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 3 – Coconut SFT
    # ───────────────────────────────────────────────────────────────────────
    config.LEARNING_RATE, config.LR_DECAY, config.DROPOUT = 1e-5, 0.95, 0.05
    for sub_epoch in range(2):
        loader = prepare_ft_dataloader(
            tokenizer, config.BLOCK_SIZE, True, args, stage=2,
            gradual_latent_mask=(sub_epoch == 0), full_latent_mask=(sub_epoch != 0)
        )
        model.train(); epoch_loss = 0.0
        for batch in loader:
            full_ids = batch["input_ids"].to(device)
            ids, offset = _seq_parallel_slice(full_ids, rank, world_size)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_coconut(
                        ids, labels=ids, offset=offset,
                        latent_token_id=model.tokenizer.convert_tokens_to_ids("<bot>")
                    )

            if world_size > 1: torch.distributed.all_reduce(loss); loss /= world_size
            if use_deepspeed:
                engine.zero_grad(); engine.backward(loss); engine.step()
            else:
                adam_optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                adam_optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"[Coconut] sub-epoch {sub_epoch+1}/2 | loss {epoch_loss/len(loader):.4f}")

    tag_dir = os.path.join(args.save_dir, "supervised_finetuned_coconut")
    os.makedirs(tag_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag="supervised_finetuned_coconut")
    else:
        torch.save(model.state_dict(), os.path.join(tag_dir, "model.pth"))

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 4 – EBM fine-tune & validation
    # ───────────────────────────────────────────────────────────────────────
    ebm_opt = torch.optim.Adam(model.ebm.parameters(), lr=args.ebm_lr)
    margin  = getattr(args, "ebm_margin", 1.0)

    for epoch in range(1, args.ebm_epochs + 1):
        model.ebm.train(); tot, steps = 0.0, 0
        for stage in range(3, 8):
            loader = prepare_ft_dataloader(tokenizer, config.BLOCK_SIZE, True, args, stage=stage)
            for batch in loader:
                ids3d   = batch["input_ids"].to(device)          # (B,K,T_full)
                ids_flat, offset = _seq_parallel_slice(ids3d, rank, world_size)
                B, K, T_local = ids_flat.shape

                with torch.no_grad():
                    emb = model.get_embeddings(ids_flat.view(B*K, T_local), pool=True, offset=offset)

                en  = model.ebm(emb).view(B, K)
                pos = en.argmin(dim=1)
                pos_en = en[torch.arange(B), pos]
                neg_en = en.masked_select(~F.one_hot(pos, K).bool()).view(B, K-1)
                loss = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()

                if world_size > 1: torch.distributed.all_reduce(loss); loss /= world_size
                ebm_opt.zero_grad(); loss.backward(); ebm_opt.step()
                tot += loss.item(); steps += 1
        logging.info(f"[EBM] epoch {epoch}/{args.ebm_epochs} | loss {tot/steps:.4f}")

    # validation
    model.ebm.eval(); val_tot = val_steps = 0
    vloader = prepare_ft_dataloader(tokenizer, config.BLOCK_SIZE, False, args, stage=8)
    with torch.no_grad():
        for batch in vloader:
            ids3d   = batch["input_ids"].to(device)
            ids_flat, offset = _seq_parallel_slice(ids3d, rank, world_size)
            B, K, T_local = ids_flat.shape

            emb = model.get_embeddings(ids_flat.view(B*K, T_local), pool=True, offset=offset)
            en  = model.ebm(emb).view(B, K)
            pos = en.argmin(dim=1)
            pos_en = en[torch.arange(B), pos]
            neg_en = en.masked_select(~F.one_hot(pos, K).bool()).view(B, K-1)
            loss = F.relu(margin + pos_en.unsqueeze(1) - neg_en).mean()

            if world_size > 1: torch.distributed.all_reduce(loss); loss /= world_size
            val_tot += loss.item(); val_steps += 1

    logging.info(f"[EBM validation] loss {val_tot/val_steps:.4f}")

    tag_dir = os.path.join(args.save_dir, "model_with_ebm")
    os.makedirs(tag_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag="model_with_ebm")
    else:
        torch.save(model.state_dict(), os.path.join(tag_dir, "model.pth"))

    logging.info("training complete.")
