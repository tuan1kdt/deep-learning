"""Ghép ImageProjection + decoder thành một model duy nhất.

Hai thí nghiệm chính chỉ khác nhau ở decoder — projection, vocab, feature
giữ nguyên — nên khác biệt kết quả quy được về kiến trúc decoder."""
from torch import nn

from final.config import Config
from final.models.image_proj import ImageProjection
from final.models.lstm_decoder import LSTMDecoder
from final.models.transformer_decoder import TransformerDecoder


class CaptionModel(nn.Module):
    def __init__(self, proj: ImageProjection, decoder: nn.Module):
        super().__init__()
        self.proj = proj
        self.decoder = decoder

    def forward(self, feats_raw, cap_in):
        return self.decoder(self.proj(feats_raw), cap_in)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_model(cfg: Config, vocab_size: int) -> CaptionModel:
    proj = ImageProjection(cfg.feat_dim, cfg.d_model, cfg.dropout)
    if cfg.decoder == "lstm":
        decoder = LSTMDecoder(vocab_size, cfg.d_model, cfg.attn_dim,
                              cfg.dropout, cfg.use_attention)
    else:
        decoder = TransformerDecoder(vocab_size, cfg.d_model, cfg.num_layers,
                                     cfg.num_heads, cfg.ffn_dim, cfg.dropout,
                                     cfg.max_len)
    return CaptionModel(proj, decoder)
