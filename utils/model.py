import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from utils.config import config
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from deepspeed.runtime.zero.stage3 import GatheredParameters
from cut_cross_entropy import linear_cross_entropy
from utils.ebm import EnergyBasedModel

# --------------------------------------------------------------------
# RoPE helpers
# --------------------------------------------------------------------
def build_sin_cos(seq_len: int, dim: int, device, base: float = 10_000.0):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float, device=device)
        * (-math.log(base) / (dim * 2))
    )
    return torch.sin(pos * div), torch.cos(pos * div)

def apply_rope(q, k, sin, cos):
    half = q.size(-1) // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

# --------------------------------------------------------------------
# Ring‐Flash MultiHeadAttention
# --------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)

        self.max_seq_len = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq_len, self.head_size // 2, "cpu")
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def update_rope_buffers(self, new_len: int, *, base: float = 500_000.0):
        self.max_seq_len = new_len
        sin, cos = build_sin_cos(new_len, self.head_size // 2, "cpu", base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x, *, return_attn_probs: bool = False):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._forward_flash(x, return_attn_probs)
        return self._forward_ring_flash(x)

    def _forward_flash(self, x, return_attn_probs: bool):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        sin, cos = self.rope_sin[:T].to(x.device), self.rope_cos[:T].to(x.device)
        q, k = apply_rope(q, k, sin, cos)

        # pack for Flash-attn
        qkv_flat = torch.stack([q, k, v], dim=2).view(B * T, 3, self.n_head, self.head_size)
        cu_seqlens = torch.arange(0, (B + 1) * T, step=T, dtype=torch.int32, device=x.device)
        out, attn_probs, _ = flash_attn_unpadded_qkvpacked_func(
            qkv_flat,
            cu_seqlens,
            T,
            dropout_p=config.DROPOUT,
            softmax_scale=None,
            causal=True,
            return_attn_probs=return_attn_probs,
        )
        out = out.view(B, T, self.n_head, self.head_size).permute(0, 2, 1, 3).reshape(B, T, C)
        proj = self.out_proj(out)
        return (proj, attn_probs, None) if return_attn_probs else proj

    def _forward_ring_flash(self, x):
        B, T_local, C = x.size()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert T_local == config.BLOCK_SIZE

        # 1) project + RoPE
        qkv = self.qkv_proj(x).view(B, T_local, 3, self.n_head, self.head_size)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        sin, cos = self.rope_sin[:T_local].to(x.device), self.rope_cos[:T_local].to(x.device)
        q, k = apply_rope(q, k, sin, cos)

        # 2) prepare buffers
        kv_local = torch.cat([k, v], dim=-1).contiguous()
        kv_recv = torch.empty_like(kv_local)
        acc = torch.zeros_like(q)

        def _flash_block(qb, kb, vb):
            # pack exactly as single‐GPU path, but causal=False
            B_, T_, H, D = qb.size()
            qkvb = torch.stack([qb, kb, vb], dim=2).view(B_ * T_, 3, H, D)
            cu = torch.arange(0, (B_ + 1) * T_, step=T_, dtype=torch.int32, device=qb.device)
            out, _, _ = flash_attn_unpadded_qkvpacked_func(
                qkvb, cu, T_,
                dropout_p=config.DROPOUT,
                softmax_scale=None,
                causal=False,
                return_attn_probs=False,
            )
            return out.view(B_, T_, H, D)

        # 3) local
        acc += _flash_block(q, k, v)

        # 4) ring
        nxt, prv = (rank + 1) % world, (rank - 1 + world) % world
        send = dist.isend(kv_local, dst=nxt)
        recv = dist.irecv(kv_recv, src=prv)
        current = kv_local

        for _ in range(world - 1):
            recv.wait()
            k_r, v_r = torch.split(kv_recv, self.head_size, dim=-1)
            acc += _flash_block(q, k_r, v_r)
            current, kv_recv = kv_recv, current
            send = dist.isend(current, dst=nxt)
            recv = dist.irecv(kv_recv, src=prv)
        send.wait()

        out = acc.permute(0, 2, 1, 3).reshape(B, T_local, C)
        return self.out_proj(out)

# --------------------------------------------------------------------
# MoE / Sparse Routing (unchanged)
# --------------------------------------------------------------------
class Expert(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), nn.Dropout(config.DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.lin1 = nn.Linear(n_embed, num_experts)
        self.lin2 = nn.Linear(n_embed, num_experts)
    def forward(self, x):
        logits = self.lin1(x)
        noise = torch.randn_like(logits) * F.softplus(self.lin2(x))
        mixed = logits + noise
        full = F.softmax(mixed, dim=-1)
        topk, idx = mixed.topk(self.top_k, dim=-1)
        sparse = torch.full_like(mixed, float("-inf")).scatter_(-1, idx, topk)
        return F.softmax(sparse, dim=-1), idx, full

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, capacity_factor=1.0):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
        self.cap_fac = capacity_factor
        self.num_experts = num_experts
    def forward(self, x):
        B, T, D = x.size()
        route, idx, full = self.router(x)
        out = torch.zeros_like(x)
        flat = x.view(-1, D)
        flatr = route.view(-1, route.size(-1))
        cap = int(B * T * self.top_k / self.num_experts * self.cap_fac)
        upd = torch.zeros_like(flat)
        for i, ex in enumerate(self.experts):
            mask = (idx == i).any(-1).view(-1)
            sel = torch.nonzero(mask).squeeze(-1)[:cap]
            if sel.numel():
                eout = ex(flat[sel])
                upd.index_add_(0, sel, eout * flatr[sel, i:i+1])
        return out + upd.view(B, T, D), (-torch.sum(full * torch.log(full + 1e-8), dim=-1)).mean() if self.training else None

class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = MultiHeadAttention(n_embed, n_head)
        self.moe = SparseMoE(n_embed, num_experts, top_k)
    def forward(self, x, mask=None):
        xm = x * mask.unsqueeze(-1) if mask is not None else x
        a = self.sa(self.ln1(xm))
        b, _ = self.moe(self.ln2(x + a))
        return x + a + b, _

# --------------------------------------------------------------------
# SparseMoE Language Model
# --------------------------------------------------------------------
class SparseMoELanguageModel(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, dropout, num_experts, top_k, tokenizer_name="hf-internal-testing/llama-tokenizer"):
        super().__init__()
        from transformers import LlamaTokenizerFast
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name, model_max_length=config.CONTEXT_WINDOW)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<bot>','<start_latent>','<end_latent>','<reasoning>','</reasoning>']})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        V = self.tokenizer.vocab_size

        self.tok_emb = nn.Embedding(V, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.token_embedding_table    = self.tok_emb
        self.position_embedding_table = self.pos_emb
        self.blocks = nn.ModuleList([Block(n_embed,n_head,num_experts,top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.attn_pool = nn.Linear(n_embed,1,bias=False)
        self.block_size = block_size
        self.ebm = EnergyBasedModel(n_embed)

    def preprocess_input(self, ids):
        B,T = ids.shape
        pad = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T>=self.block_size: return ids[:,-self.block_size:]
        p = torch.full((B,self.block_size-T),pad,dtype=ids.dtype,device=ids.device)
        return torch.cat([ids,p],dim=1)

    def forward_next_token_efficient(self, input_ids, reduction="mean", attention_mask=None, force_bf16=False):
        device = input_ids.device
        ids = self.preprocess_input(input_ids)
        T = self.block_size
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T,device=device))
        for blk in self.blocks:
            x,_ = blk(x, attention_mask)
        x = self.ln_f(x)
        if force_bf16: x = x.to(torch.bfloat16)
        with GatheredParameters(self.tok_emb.weight,modifier_rank=0):
            W = getattr(self,"_gathered_weights",None) or self.tok_emb.weight.clone().to(device)
        loss = linear_cross_entropy(x, W, ids, ignore_index=int(self.tokenizer.pad_token_id),reduction=reduction,shift=1)
        return loss

    def get_embeddings(self, input_ids, pool=False, attention_mask=None):
        device = input_ids.device
        ids = self.preprocess_input(input_ids)
        if attention_mask is None:
            pad = int(self.tokenizer.pad_token_id)
            attention_mask = (ids!=pad).to(device)
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(self.block_size,device=device))
        for blk in self.blocks:
            x,_ = blk(x, attention_mask)
        x = self.ln_f(x)
        if pool:
            scores = self.attn_pool(x).squeeze(-1).masked_fill(~attention_mask, -1e9)
            w = F.softmax(scores,dim=1)
            return torch.einsum("btd,bt->bd", x, w)
        return x

    def forward_coconut(self, input_ids, attention_mask=None, labels=None, latent_token_id=99998, reduction="mean", force_bf16=False):
        device = input_ids.device
        ids = self.preprocess_input(input_ids)
        pad = int(self.tokenizer.pad_token_id)
        if attention_mask is None:
            attention_mask = (ids!=pad).to(device)
        B,T = ids.shape
        embeds = (self.tok_emb(ids) + self.pos_emb(torch.arange(T,device=device))).detach().requires_grad_(True)
        lat_positions = [(ids[b]==latent_token_id).nonzero(as_tuple=True)[0].tolist() for b in range(B)]
        max_lat = max((len(l) for l in lat_positions), default=0)
        for idx in range(max_lat):
            seq_len = min(l[idx] for l in lat_positions if idx<len(l))+1
            x = embeds[:,:seq_len,:]
            m = attention_mask[:,:seq_len]
            for blk in self.blocks:
                x,_ = blk(x,m)
            x = self.ln_f(x)
            if force_bf16: x = x.to(torch.bfloat16)
            for b,l in enumerate(lat_positions):
                if idx<len(l) and l[idx]>0:
                    embeds[b,l[idx]] = x[b,l[idx]-1]
        x = embeds
        for blk in self.blocks:
            x,_ = blk(x, attention_mask)
        x = self.ln_f(x)
        if labels is None: return None
        with GatheredParameters(self.tok_emb.weight,modifier_rank=0):
            W = getattr(self,"_gathered_weights",None) or self.tok_emb.weight.clone().to(device)
        if force_bf16: W = W.to(torch.bfloat16)
        return linear_cross_entropy(x, W, labels[:,-T:], ignore_index=pad, reduction=reduction, shift=1)

def update_model_rope_for_extended_context(model, new_seq_len, base=500_000.0):
    for blk in model.blocks:
        blk.sa.update_rope_buffers(new_seq_len, base=base)
    return model

def expand_pos_embedding(model, new_len):
    old_len, dim = model.pos_emb.weight.shape
    if new_len <= old_len:
        model.block_size = new_len
        return
    new_emb = nn.Embedding(new_len, dim, device=model.pos_emb.weight.device)
    new_emb.weight.data[:old_len] = model.pos_emb.weight.data
    nn.init.normal_(new_emb.weight.data[old_len:], std=0.02)
    model.pos_emb = new_emb
    model.position_embedding_table = new_emb
    model.block_size = new_len
