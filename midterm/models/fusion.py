"""Ba chiến lược fusion — trục ablation chính của đồ án.

Interface chung: forward(v_img, img_map, v_txt) → (fused, attn_weights | None)
- v_img   (B, d): vector ảnh toàn cục
- img_map (B, 49, d): 49 vùng không gian của ảnh (chỉ cross_attention dùng)
- v_txt   (B, d): vector câu hỏi
- fused   (B, d): vector hợp nhất đưa vào MLP head
- attn_weights (B, 49): chỉ có ở cross_attention — để vẽ heatmap minh họa.

Lưu ý diễn giải ablation: cross_attention nhận thêm thông tin spatial (img_map)
mà concat/hadamard không dùng — khác biệt kết quả gộp cả "cơ chế fusion" lẫn
"có thông tin spatial" (xem spec mục 3.3). Báo cáo phải nêu confound này.
"""
import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Baseline đơn giản nhất: nối [v_img ; v_txt] rồi chiếu tuyến tính + ReLU."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.fc = nn.Linear(2 * d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, v_img, img_map, v_txt):
        return self.relu(self.fc(torch.cat([v_img, v_txt], dim=-1))), None


class HadamardFusion(nn.Module):
    """Tương tác nhân: chiếu mỗi modality qua Linear riêng rồi nhân từng phần tử.
    Phép nhân buộc hai modality "đồng thuận" theo từng chiều — feature chỉ lớn
    khi cả ảnh lẫn câu hỏi cùng kích hoạt chiều đó."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.proj_img = nn.Linear(d_model, d_model)
        self.proj_txt = nn.Linear(d_model, d_model)

    def forward(self, v_img, img_map, v_txt):
        return self.proj_img(v_img) * self.proj_txt(v_txt), None


class CrossAttentionFusion(nn.Module):
    """Câu hỏi (query duy nhất) "nhìn" vào 49 vùng ảnh (key/value) bằng
    multi-head attention, cộng residual với v_txt rồi LayerNorm — một block
    Transformer tối giản. Attention weights cho biết model nhìn vào đâu."""

    def __init__(self, d_model: int = 768, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, v_img, img_map, v_txt):
        query = v_txt.unsqueeze(1)                              # (B, 1, d)
        attended, weights = self.attn(query, img_map, img_map)  # (B,1,d), (B,1,49)
        fused = self.norm(v_txt + attended.squeeze(1))          # residual + LN
        return fused, weights.squeeze(1)                        # attn: (B, 49)


def build_fusion(cfg):
    """Factory: đổi fusion = đổi config, không đụng phần code khác."""
    if cfg.fusion == "concat":
        return ConcatFusion(cfg.d_model)
    if cfg.fusion == "hadamard":
        return HadamardFusion(cfg.d_model)
    return CrossAttentionFusion(cfg.d_model, cfg.num_heads)
