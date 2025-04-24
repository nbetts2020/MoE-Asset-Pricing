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

def extract_label_value(decoded_text):
    """
    Extracts the numerical label following the "<30 DAY LABEL>:" marker and returns it as a float.
    If multiple dots are found (e.g. "2..98"), they are replaced with a single dot.
    Returns None if extraction or conversion fails.
    """
    match = re.search(r'\<30 DAY LABEL\>:\s*([\d\.]+)', decoded_text)
    if match:
        num_str = match.group(1)
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
    optimizer,      # Single optimizer for main model
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
    Training loop:
      1. Normal pretraining (ft_dataset_1)
      2. Continual pretraining with 64k context (ft_dataset_1)
      3. Supervised fine-tuning with Coconut (ft_dataset_2)
      4. EBM fine-tuning on bootstrap datasets (ft_dataset_3-7) + validation on ft_dataset_8

    If args.stage_1_only == True, only phase 1 is run and the function returns immediately after saving.
    """
    if use_deepspeed:
        engine = model
    else:
        adam_optimizer = optimizer

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # ------------------------------------------------------------------------
    # PHASE 1: Normal Pretraining (ft_dataset_1)
    # ------------------------------------------------------------------------
    logging.info(f"Starting normal pretraining on rank {rank}.")
    logging.info(f"Dataloader length: {len(dataloader)} batches.")
    PRETRAIN_EPOCHS = epochs
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            print(step, len(dataloader), "progress!!")
            input_ids = batch['input_ids'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_next_token_efficient(input_ids, reduction="mean")

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()

            # online learning penalties
            if args.use_si and si:
                si.update_weights(model)
            if args.use_ewc and ewc:
                for ewc_inst in ewc:
                    loss = loss + args.lambda_ewc * ewc_inst.penalty(model)
            if replay_buffer and len(replay_buffer.buffer) > 0:
                replay_loss = replay_buffer.replay_and_calculate_loss(
                    model=model,
                    tokenizer=tokenizer,
                    replay_batch_size=args.replay_batch_size,
                    device=device,
                    alpha=args.replay_buffer_weight
                )
                loss = loss + replay_loss

            epoch_loss += loss.item()
        logging.info(f"[Normal Pretrain] Epoch {epoch}/{PRETRAIN_EPOCHS}, Avg Loss: {epoch_loss/len(dataloader):.4f}")

    # save checkpoint...
    pretrain_tag = "normal_pretrained"
    pretrain_dir = os.path.join(args.save_dir, pretrain_tag)
    os.makedirs(pretrain_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=pretrain_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(pretrain_dir, "model.pth"))

    # if only doing stage 1, exit here
    if getattr(args, "stage_1_only", False):
        logging.info("stage_1_only=True, exiting after phase 1 pretraining.")
        return

    # ------------------------------------------------------------------------
    # PHASE 2: Continual Pretraining (64k context ft_dataset_1)
    # ------------------------------------------------------------------------
    logging.info("=== Starting Continual Pretraining with Extended Context ===")
    config.BLOCK_SIZE = 65536
    config.CONTEXT_WINDOW = 65536
    model.tokenizer.model_max_length = config.CONTEXT_WINDOW
    expand_pos_embedding(model, new_seq_len=config.BLOCK_SIZE)
    update_model_rope_for_extended_context(model, new_seq_len=config.BLOCK_SIZE, base=500000.0)

    from utils.utils import prepare_optimizer
    new_opts = prepare_optimizer(model, args)
    if use_deepspeed:
        engine.optimizer = new_opts["main"]
    else:
        adam_optimizer  = new_opts["main"]

    continual_loader = prepare_ft_dataloader(
        tokenizer,
        block_size=config.BLOCK_SIZE,
        shuffle=False,
        args=args,
        stage=1
    )
    for epoch in range(1, 2):
        model.train()
        epoch_loss = 0.0
        for batch in continual_loader:
            input_ids = batch['input_ids'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_next_token_efficient(input_ids, reduction="mean")

            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()

            epoch_loss += loss.item()
        logging.info(f"[Continual] Epoch {epoch}/1, Avg Loss: {epoch_loss/len(continual_loader):.4f}")

    # save checkpoint...
    continual_tag = "continual_pretrained_64k"
    continual_dir = os.path.join(args.save_dir, continual_tag)
    os.makedirs(continual_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=continual_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(continual_dir, "model.pth"))

    # ------------------------------------------------------------------------
    # PHASE 3: Supervised Fine-Tuning with Coconut (ft_dataset_2)
    # ------------------------------------------------------------------------
    logging.info("=== Starting Supervised Fine-Tuning (Coconut) ===")
    config.LEARNING_RATE = 1e-5
    config.LR_DECAY = 0.95
    config.DROPOUT = 0.05

    for sub_epoch in range(2):
        gradual = (sub_epoch == 0)
        ft_loader = prepare_ft_dataloader(
            tokenizer,
            block_size=config.BLOCK_SIZE,
            shuffle=True,
            args=args,
            stage=2,
            gradual_latent_mask=gradual,
            full_latent_mask=not gradual
        )
        model.train()
        epoch_loss = 0.0
        for batch in ft_loader:
            input_ids = batch['input_ids'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                with GatheredParameters(model.token_embedding_table.weight, modifier_rank=0):
                    model._gathered_weights = model.token_embedding_table.weight.clone().half()
                    loss = model.forward_coconut(
                        input_ids=input_ids,
                        attention_mask=None,
                        labels=input_ids,
                        latent_token_id=model.tokenizer.convert_tokens_to_ids("<bot>"),
                        reduction="mean"
                    )
            if use_deepspeed:
                engine.zero_grad()
                engine.backward(loss)
                engine.step()
            else:
                adam_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                adam_optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"[Coconut] Sub-epoch {sub_epoch+1}/2, Avg Loss: {epoch_loss/len(ft_loader):.4f}")

    # save checkpoint...
    coconut_tag = "supervised_finetuned_coconut"
    coconut_dir = os.path.join(args.save_dir, coconut_tag)
    os.makedirs(coconut_dir, exist_ok=True)
    if use_deepspeed:
        model.save_checkpoint(args.save_dir, tag=coconut_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(coconut_dir, "model.pth"))

    # ------------------------------------------------------------------------
    # PHASE 4: EBM Fine-Tuning (ft_dataset_3-7) + Validation (ft_dataset_8)
    # ------------------------------------------------------------------------
    logging.info("=== Starting EBM Fine-Tuning Phase ===")
    ebm_optimizer = torch.optim.Adam(model.ebm.parameters(), lr=args.ebm_lr)
    margin = getattr(args, "ebm_margin", 1.0)

    for epoch in range(1, args.ebm_epochs + 1):
        model.ebm.train()
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
                B, K, T = ids.shape
                flat_ids = ids.view(B*K, T)

                with torch.no_grad():
                    embs = model.get_embeddings(flat_ids, pool=True)
                embs = embs.view(B, K, -1)

                flat_embs = embs.view(B*K, -1)
                energies = model.ebm(flat_embs).view(B, K)

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
    model.ebm.eval()
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
            B, K, T = ids.shape
            flat_ids = ids.view(B*K, T)
            embs = model.get_embeddings(flat_ids, pool=True).view(B, K, -1)
            energies = model.ebm(embs.view(B*K, -1)).view(B, K)
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
        model.save_checkpoint(args.save_dir, tag=final_tag, client_state={})
    else:
        torch.save(model.state_dict(), os.path.join(final_dir, "model_with_ebm.pth"))

    logging.info("Training complete.")
