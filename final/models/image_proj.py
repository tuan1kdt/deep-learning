"""Chiếu 49 vùng ResNet (2048-d) về d_model — thành phần dùng chung cho cả
hai decoder, tách riêng để so sánh công bằng: khác nhau CHỈ ở decoder."""
from torch import nn


class ImageProjection(nn.Module):
    def __init__(self, feat_dim: int = 2048, d_model: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, feats):  # (B, R, feat_dim) → (B, R, d_model)
        return self.drop(self.norm(self.proj(feats)))
