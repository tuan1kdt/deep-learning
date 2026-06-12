"""VQAModel: ghép 4 thành phần theo config — image encoder + text encoder
+ fusion + MLP head. Encoder freeze nên phần trainable chỉ vài triệu tham số,
train được trên T4 trong vài phút mỗi epoch.
"""
import torch
from torch import nn

from midterm.models.fusion import build_fusion
from midterm.models.image_encoder import ImageEncoder
from midterm.models.text_encoder import TextEncoder


class VQAModel(nn.Module):
    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg.d_model, cfg.unfreeze_last_block)
        self.text_encoder = TextEncoder(cfg.text_model_name, cfg.text_pool)
        self.fusion = build_fusion(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),  # 0.5 — chống overfit trên 1.6k mẫu train
            nn.Linear(cfg.hidden_dim, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        v_img, img_map = self.image_encoder(images)
        v_txt = self.text_encoder(input_ids, attention_mask)
        fused, attn = self.fusion(v_img, img_map, v_txt)
        return self.head(fused), attn

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return total, trainable
