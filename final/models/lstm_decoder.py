"""LSTM decoder kiểu Show, Attend and Tell: mỗi bước sinh từ, attention
Bahdanau nhìn lại 49 vùng ảnh. Đây là bản PyTorch của đúng cơ chế đã tự cài
bằng NumPy ở day6 (seq2seq + attention) — báo cáo sẽ nêu liên hệ này.

Biến thể use_attention=False (ablation #2): thay context mỗi bước bằng
mean-pool cố định của 49 vùng — đo xem attention đóng góp bao nhiêu.
"""
import torch
from torch import nn


class BahdanauAttention(nn.Module):
    """score(h, f_i) = v^T tanh(W_h h + W_f f_i) — attention cộng tính."""

    def __init__(self, d_model: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(d_model, attn_dim)
        self.W_f = nn.Linear(d_model, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, h, feats):  # h (B,D), feats (B,R,D)
        scores = self.v(torch.tanh(self.W_f(feats)
                                   + self.W_h(h).unsqueeze(1))).squeeze(-1)
        weights = scores.softmax(dim=-1)                    # (B,R)
        context = (weights.unsqueeze(-1) * feats).sum(dim=1)  # (B,D)
        return context, weights


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, attn_dim: int = 512,
                 dropout: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Embedding init N(0,1) mặc định quá lớn khi tie weight với output layer
        # (sau LayerNorm chuẩn đơn vị, logits có std ~ sqrt(d_model) → CE ban đầu ~267
        # thay vì ln(V)≈7.8 — phát hiện qua smoke run). Init std=0.02 kiểu GPT.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()  # giữ hàng padding_idx=0 bằng 0
        self.attention = BahdanauAttention(d_model, attn_dim) if use_attention else None
        # input mỗi bước = [embedding từ trước ; context ảnh] → 2*d_model
        self.cell = nn.LSTMCell(2 * d_model, d_model)
        self.init_h = nn.Linear(d_model, d_model)
        self.init_c = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # Tie weight: output layer dùng chính ma trận embedding — giảm ~1.5M
        # tham số và buộc không gian vào/ra của từ nhất quán.
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.fc.weight = self.embedding.weight

    def forward(self, feats_proj, cap_in):
        B, T = cap_in.shape
        R = feats_proj.size(1)
        mean_f = feats_proj.mean(dim=1)
        h = torch.tanh(self.init_h(mean_f))
        c = torch.tanh(self.init_c(mean_f))
        emb = self.drop(self.embedding(cap_in))             # (B,T,D)

        logits, attns = [], []
        uniform = torch.full((B, R), 1.0 / R, device=feats_proj.device)
        for t in range(T):
            if self.use_attention:
                # attention dùng h của bước TRƯỚC — quyết định "nhìn đâu"
                # trước khi đọc từ mới
                context, w = self.attention(h, feats_proj)
            else:
                context, w = mean_f, uniform
            h, c = self.cell(torch.cat([emb[:, t], context], dim=-1), (h, c))
            logits.append(self.fc(self.drop(h)))
            attns.append(w)
        return torch.stack(logits, dim=1), torch.stack(attns, dim=1)
