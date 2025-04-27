# utils/model.py
# =============================================================================
#  Ring-Flash-Attention Sparse-MoE LM
# =============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.config import config
from utils.ebm    import EnergyBasedModel
from cut_cross_entropy import linear_cross_entropy
from deepspeed.runtime.zero.stage3 import GatheredParameters

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_unpadded_qkvpacked_func,
    flash_attn_unpadded_kvpacked_func as flash_attn_kvpacked_func,
)
from torch.distributed.nn.functional import broadcast

def build_sin_cos(seq_len: int, dim: int, device, base: float = 10_000.0):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    inv = torch.exp(
        torch.arange(0, dim * 2, 2, dtype=torch.float, device=device)
        * (-math.log(base) / (dim * 2))
    )
    return torch.sin(pos * inv), torch.cos(pos * inv)

def apply_rope(q, k, sin, cos):
    half = q.size(-1) // 2
    if sin.size(-1) != half:
        sin, cos = sin[..., :half], cos[..., :half]
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    q1, q2 = q.split(half, dim=-1)
    k1, k2 = k.split(half, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head, self.head_dim = n_head, n_embed // n_head

        self.qkv  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

        self.max_seq = config.BLOCK_SIZE
        sin, cos = build_sin_cos(self.max_seq, self.head_dim // 2, "cpu")
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def update_rope_buffers(self, new_len: int, base: float = 5e5):
        self.max_seq = new_len
        device = self.rope_sin.device if hasattr(self, "rope_sin") else "cpu"
        sin, cos = build_sin_cos(new_len, self.head_dim // 2, device, base)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("rope_cos", cos, persistent=False)

    def forward(self, x, *, return_attn_probs=False):
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._forward_flash(x, return_attn_probs)
        return self._forward_ring_flash(x, return_attn_probs)

    def _forward_flash(self, x, return_attn_probs=False):
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        sin_t = self.rope_sin[:T].to(x.device)
        cos_t = self.rope_cos[:T].to(x.device)
        q, k = apply_rope(q, k, sin_t, cos_t)

        qkv_flat  = torch.stack([q, k, v], dim=2).view(B * T, 3, self.n_head, self.head_dim)
        cu_seqlens = torch.arange(0, (B + 1) * T, T, device=x.device, dtype=torch.int32)
        out = flash_attn_unpadded_qkvpacked_func(
            qkv_flat, cu_seqlens, T,
            dropout_p=config.DROPOUT, softmax_scale=None,
            causal=True, return_attn_probs=return_attn_probs,
        )
        if return_attn_probs:
            proj_out, attn_probs = out[0], out[1]
        else:
            proj_out, attn_probs = out, None
        proj_out = proj_out.view(B, T, self.n_head, self.head_dim).permute(0,2,1,3).reshape_as(x)
        return (self.proj(proj_out), attn_probs, None) if return_attn_probs else self.proj(proj_out)

    def _forward_ring_flash(self, x, return_attn_probs=False):
        B, T_local, C = x.shape
        device = x.device
        rank = dist.get_rank()
        world = dist.get_world_size()
        T_global = T_local * world

        # ensure RoPE covers global positions
        if self.rope_sin.size(0) < T_global:
            self.update_rope_buffers(T_global)

        # shard input
        qkv = self.qkv(x).view(B, T_local, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        start = rank * T_local
        sin_t = self.rope_sin[start:start+T_local].to(device)
        cos_t = self.rope_cos[start:start+T_local].to(device)
        q, k = apply_rope(q, k, sin_t, cos_t)

        q, k, v = q.half(), k.half(), v.half()
        cu_seqlens = torch.arange(0, (B+1)*T_local, T_local, device=device, dtype=torch.int32)
        acc = torch.zeros_like(q)
        kv_cache = [k.clone(), v.clone()]

        for step in range(world):
            curr_k, curr_v = kv_cache
            max_k = T_local * (step + 1)
            q_flat  = q.reshape(B*T_local, self.n_head, self.head_dim)
            kv_flat = torch.stack([curr_k, curr_v], dim=1).reshape(B*max_k, 2, self.n_head, self.head_dim)
            out = flash_attn_kvpacked_func(
                q_flat, kv_flat,
                cu_seqlens, torch.arange(0, (B+1)*max_k, max_k, device=device, dtype=torch.int32),
                T_local, max_k,
                dropout_p=config.DROPOUT, softmax_scale=None,
                causal=True,
            )
            proj_out = out[0] if return_attn_probs else out
            acc += proj_out.view(B, T_local, self.n_head, self.head_dim)

            if step < world - 1:
                next_rank = (rank + 1) % world
                prev_rank = (rank - 1) % world
                send_k, send_v = k.clone(), v.clone()
                rcv_k = torch.empty_like(k)
                rcv_v = torch.empty_like(v)
                reqs = [
                    dist.isend(send_k, next_rank),
                    dist.isend(send_v, next_rank),
                    dist.irecv(rcv_k, prev_rank),
                    dist.irecv(rcv_v, prev_rank)
                ]
                for r in reqs: r.wait()
                kv_cache = [torch.cat([curr_k, rcv_k], dim=1),
                            torch.cat([curr_v, rcv_v], dim=1)]

        out = acc.reshape(B, T_local, C)
        proj = self.proj(out)
        return (proj, out if return_attn_probs else None, None) if return_attn_probs else proj

class Expert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4*d), nn.ReLU(),
            nn.Linear(4*d, d), nn.Dropout(config.DROPOUT),
        )
    def forward(self, x): return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self, d, n_exp, top_k):
        super().__init__()
        self.top_k = top_k
        self.lin_w = nn.Linear(d, n_exp)
        self.lin_n = nn.Linear(d, n_exp)
    def forward(self, h):
        logits = self.lin_w(h)
        noise  = torch.randn_like(logits) * F.softplus(self.lin_n(h))
        noisy  = logits + noise
        full   = F.softmax(noisy, -1)
        topk, ix = noisy.topk(self.top_k, -1)
        sparse = torch.full_like(noisy, -float("inf")).scatter(-1, ix, topk)
        return F.softmax(sparse, -1), ix, full

class SparseMoE(nn.Module):
    def __init__(self, d, n_exp, top_k, cap=1.0):
        super().__init__()
        self.router = NoisyTopkRouter(d, n_exp, top_k)
        self.experts= nn.ModuleList([Expert(d) for _ in range(n_exp)])
        self.top_k, self.cap, self.n_exp = top_k, cap, n_exp

    def forward(self, x):
        B,T,D = x.shape
        p, ix, full = self.router(x)
        flat_x = x.view(-1,D); flat_p = p.view(-1,self.n_exp)
        upd = torch.zeros_like(flat_x)
        cap = int(B*T*self.top_k/self.n_exp*self.cap)
        for i, exp in enumerate(self.experts):
            mask = (ix==i).any(-1).view(-1)
            sel  = torch.nonzero(mask).squeeze(-1)[:cap]
            if sel.numel():
                o = exp(flat_x[sel])
                upd.index_add_(0, sel, o*flat_p[sel,i:i+1])
        y = upd.view(B,T,D)
        ent = (-full*full.clamp_min(1e-8).log()).sum(-1).mean() if self.training else None
        return y, ent

class Block(nn.Module):
    def __init__(self, d, n_head, n_exp, top_k):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.sa  = MultiHeadAttention(d, n_head)
        self.moe = SparseMoE(d, n_exp, top_k)
    def forward(self, x, mask=None):
        x_mask = x if mask is None else x*mask.unsqueeze(-1)
        a = self.sa(self.ln1(x_mask))
        y,_ = self.moe(self.ln2(x+a))
        return x + a + y, _

class SparseMoELanguageModel(nn.Module):
    def __init__(self,
                 n_embed, n_head, n_layer, block_size,
                 dropout, num_experts, top_k,
                 tokenizer_name="hf-internal-testing/llama-tokenizer"):
        super().__init__()
        from transformers import LlamaTokenizerFast
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            tokenizer_name, model_max_length=config.BLOCK_SIZE
        )
        self.tokenizer.add_special_tokens({
            "additional_special_tokens":[
                "<bot>","<start_latent>","<end_latent>",
                "<reasoning>","</reasoning>"
            ]
        })
        self.tokenizer.pad_token = self.tokenizer.eos_token
        V = self.tokenizer.vocab_size

        self.tok_emb = nn.Embedding(V, n_embed)
        self.pos_emb = nn.Embedding(block_size, n_embed)
        self.token_embedding_table    = self.tok_emb
        self.position_embedding_table = self.pos_emb

        self.blocks = nn.ModuleList([
            Block(n_embed, n_head, num_experts, top_k)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embed)
        self.attn_pool = nn.Linear(n_embed, 1, bias=False)
        self.block_size = block_size
        self.ebm = EnergyBasedModel(n_embed)

    def _pad_or_trim(self, ids):
        B,T = ids.shape
        pad = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if T>=self.block_size: return ids[:,-self.block_size:]
        p = torch.full((B,self.block_size-T), pad, dtype=ids.dtype, device=ids.device)
        return torch.cat([ids,p],1)

    def forward_next_token_efficient(
        self, input_ids, reduction="mean",
        attention_mask=None, force_bf16=False
    ):
        ids = self._pad_or_trim(input_ids)
        B,T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T,device=ids.device))
        for blk in self.blocks:
            x,_ = blk(x, attention_mask)
        x = self.ln_f(x)
        if force_bf16: x = x.to(torch.float16)

        with GatheredParameters(self.tok_emb.weight, modifier_rank=0):
            W = getattr(self,"_gathered_weights",None) or self.tok_emb.weight.to(ids.device)

        pad = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return linear_cross_entropy(
            x, W, ids, ignore_index=pad, reduction=reduction, shift=1
        )

    def get_embeddings(self, input_ids, pool=False, attention_mask=None):
        ids = self._pad_or_trim(input_ids)
        if attention_mask is None:
            pad = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            attention_mask = (ids!=pad).to(ids.device)
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(self.block_size,device=ids.device))
        for blk in self.blocks:
            x,_ = blk(x, attention_mask)
        x = self.ln_f(x)
        if not pool: return x
        scores = self.attn_pool(x).squeeze(-1).masked_fill(~attention_mask,-1e9)
        weights=F.softmax(scores,1)
        return torch.einsum("btd,bt->bd",x,weights)

    def forward_coconut(
        self, input_ids, attention_mask=None, labels=None,
        latent_token_id=99998, reduction="mean", force_bf16=False
    ):
        ids = self._pad_or_trim(input_ids)
        pad = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if attention_mask is None:
            attention_mask=(ids!=pad).to(ids.device)
        B,T = ids.shape
        base = (self.tok_emb(ids)+
                self.pos_emb(torch.arange(T,device=ids.device))).detach()
        embeds = base.clone().requires_grad_(True)

        lat_pos = [(ids[b]==latent_token_id).nonzero(as_tuple=True)[0].tolist()
                   for b in range(B)]
        max_lat = max((len(p) for p in lat_pos), default=0)

        for p in range(max_lat):
            active=[lst[p] for lst in lat_pos if p<len(lst)]
            if not active: continue
            cut = min(active)+1
            x = embeds[:,:cut,:]; m=attention_mask[:,:cut]
            for blk in self.blocks: x,_=blk(x,m)
            x=self.ln_f(x)
            if force_bf16: x=x.to(torch.float16)
            for b,lst in enumerate(lat_pos):
                if p<len(lst) and lst[p]>0:
                    embeds[b,lst[p]] = x[b,lst[p]-1]

        x=embeds
        for blk in self.blocks: x,_=blk(x,attention_mask)
        x=self.ln_f(x)
        if force_bf16: x=x.to(torch.float16)

        if labels is None: return x
        labels=labels[:,-T:]
        with GatheredParameters(self.tok_emb.weight,modifier_rank=0):
            W=getattr(self,"_gathered_weights",None) or self.tok_emb.weight.to(ids.device)
        return linear_cross_entropy(
            x, W, labels, ignore_index=pad, reduction=reduction, shift=1
        )

def update_model_rope_for_extended_context(model, new_len, base: float = 5e5):
    for blk in model.blocks:
        blk.sa.update_rope_buffers(new_len, base)
    return model

def expand_pos_embedding(model, new_len):
    old,dim = model.pos_emb.weight.shape
    if new_len<=old:
        model.block_size=new_len
        return
    new_emb=nn.Embedding(new_len,dim,device=model.pos_emb.weight.device)
    new_emb.weight.data[:old]=model.pos_emb.weight.data
    nn.init.normal_(new_emb.weight.data[old:],std=0.02)
    model.pos_emb=new_emb
    model.position_embedding_table=new_emb
    model.block_size=new_len
