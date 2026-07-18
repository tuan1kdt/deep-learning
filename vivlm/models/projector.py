"""Pixel-shuffle 196->49 token ảnh + MLP chiếu vào không gian embedding GPT.

Gom 2x2 patch lân cận thành 1 token (channel x4) — giảm 4 lần chi phí attention
cho token ảnh mà không vứt thông tin (nanoVLM/InternVL cùng cách).
"""
import torch.nn as nn
import torch.nn.functional as F


def pixel_shuffle(x, grid=14, scale=2):
    B, N, C = x.shape
    assert N == grid * grid
    g = grid // scale
    x = x.view(B, g, scale, g, scale, C).permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, g * g, scale * scale * C)


class PixelShuffleProjector(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, grid=14, scale=2):
        super().__init__()
        self.grid, self.scale = grid, scale
        self.fc1 = nn.Linear(in_dim * scale * scale, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = pixel_shuffle(x, self.grid, self.scale)
        return self.fc2(F.silu(self.fc1(x)))
