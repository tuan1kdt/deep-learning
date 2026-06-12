"""Text encoder: BERT pretrained, freeze toàn bộ.

Vector câu hỏi 768-d lấy theo config text_pool:
- "mean" (mặc định): mean-pooling hidden state lớp cuối theo attention mask.
  [CLS] của BERT freeze vốn được pretrain cho next-sentence prediction nên là
  biểu diễn câu yếu — mean-pooling thường tốt hơn rõ rệt khi không fine-tune.
- "cls": embedding [CLS] — giữ làm đối chứng (thí nghiệm phụ).
"""
import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pool: str = "mean"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pool = pool
        for p in self.bert.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # Freeze hoàn toàn → luôn eval: tắt dropout bên trong BERT để biểu diễn
        # câu hỏi ổn định giữa các batch (cùng lý do BN ở image encoder).
        super().train(mode)
        self.bert.eval()
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                   # (B, L, 768)
        if self.pool == "cls":
            return hidden[:, 0]
        # mean-pooling: chỉ trung bình trên token thật (mask=1), bỏ padding
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
