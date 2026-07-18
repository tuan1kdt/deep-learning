"""GPT decoder-only Llama-style: RoPE + RMSNorm + SwiGLU, không bias, tied embedding.

Khác GPT-2 nguyên bản ở 3 điểm (chuẩn thực hành 2026):
- RoPE thay learned positional embedding: mã hóa vị trí TƯƠNG ĐỐI bằng phép xoay q/k.
- RMSNorm thay LayerNorm: bỏ mean-centering, rẻ hơn, ổn định tương đương.
- SwiGLU thay GELU-MLP: gate nhân điểm, chất lượng/FLOP tốt hơn.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vivlm.config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x.float() * rms).type_as(x)


def precompute_rope(head_dim: int, context: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(context).float()
    freqs = torch.outer(t, inv_freq)          # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd) — xoay từng cặp chiều (2i, 2i+1) một góc ~ vị trí
    T = x.size(2)
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight        # weight tying
        cos, sin = precompute_rope(cfg.d_model // cfg.n_head, cfg.context,
                                   cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # residual projection scale 1/sqrt(2*n_layer) (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weight đếm 1 lần

    def forward_from_embeds(self, x, targets=None):
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None      # inference: chỉ vị trí cuối
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def forward(self, idx, targets=None):
        return self.forward_from_embeds(self.tok_emb(idx), targets)

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas,
                                 fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=None,
                 eos_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.context:])
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_p is not None:
                    sl, si = torch.sort(logits, descending=True)
                    probs = F.softmax(sl, dim=-1)
                    mask = probs.cumsum(-1) - probs > top_p
                    sl[mask] = float("-inf")
                    logits = torch.full_like(logits, float("-inf")) \
                                  .scatter(1, si, sl)
                next_id = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
