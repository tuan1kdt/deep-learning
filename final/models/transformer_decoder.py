"""Transformer decoder tự cài từng block (pre-norm) thay vì nn.TransformerDecoder.

Lý do tự cài: nn.TransformerDecoder không trả cross-attention weights ra ngoài,
mà heatmap "từ nào nhìn vùng ảnh nào" là hình quan trọng nhất của báo cáo.
Mỗi block: masked self-attention (text tự nhìn quá khứ của nó — vai trò "text
encoder" trong sơ đồ) → cross-attention vào 49 vùng ảnh (chính là FUSION bằng
attention trên sơ đồ của giảng viên) → FFN.
"""
import torch
from torch import nn


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask):
        # Pre-norm: norm trước attention — ổn định hơn hẳn post-norm khi
        # train từ đầu trên data nhỏ, không cần warmup dài.
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.drop(a)
        h = self.norm2(x)
        a, w = self.cross_attn(h, memory, memory,
                               need_weights=True, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, w  # w: (B, T, R) — trung bình các head


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 3,
                 num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.3,
                 max_len: int = 22):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)  # bắt buộc với pre-norm
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight

    def forward(self, feats_proj, cap_in):
        B, T = cap_in.shape
        assert T <= self.max_len, f"chuỗi dài {T} > max_len {self.max_len}"
        pos = torch.arange(T, device=cap_in.device)
        x = self.drop(self.embedding(cap_in) + self.pos_embedding(pos))
        # True = bị chặn: tam giác trên (không nhìn tương lai)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                       device=cap_in.device), diagonal=1)
        attn = None
        for block in self.blocks:
            x, attn = block(x, feats_proj, causal)  # giữ attn của tầng CUỐI
        return self.fc(self.final_norm(x)), attn
