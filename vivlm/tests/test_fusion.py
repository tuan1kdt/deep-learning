import torch
import torch.nn as nn

from vivlm.config import GPTConfig
from vivlm.models.gpt import GPT
from vivlm.models.projector import PixelShuffleProjector, pixel_shuffle
from vivlm.models.vivlm import ViVLM

TINY = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32,
                 context=128, mlp_hidden=64)


class StubEncoder(nn.Module):
    """Giả SigLIP: (B,3,224,224) -> (B,196,768) hằng số."""
    def forward(self, pixel_values):
        return torch.zeros(pixel_values.size(0), 196, 768)


def _vlm():
    gpt = GPT(TINY)
    proj = PixelShuffleProjector(in_dim=768, out_dim=TINY.d_model)
    return ViVLM(gpt, StubEncoder(), proj)


def test_pixel_shuffle_shape_and_grouping():
    x = torch.arange(196).float().view(1, 196, 1).expand(1, 196, 4)
    y = pixel_shuffle(x, grid=14, scale=2)
    assert y.shape == (1, 49, 16)
    # token đầu ra 0 gom đúng 4 patch góc trên trái: (0,0),(0,1),(1,0),(1,1)
    got = set(y[0, 0].unique().tolist())
    assert got == {0.0, 1.0, 14.0, 15.0}


def test_projector_shape():
    p = PixelShuffleProjector(in_dim=768, out_dim=32)
    assert p(torch.randn(2, 196, 768)).shape == (2, 49, 32)


def test_vivlm_forward_loss():
    m = _vlm()
    px = torch.randn(2, 3, 224, 224)
    ids = torch.randint(0, 64, (2, 10))
    labels = torch.randint(0, 64, (2, 10))
    logits, loss = m(px, ids, labels)
    assert logits.shape == (2, 49 + 10, 64) and loss.isfinite()


def test_vivlm_loss_masking():
    m = _vlm()
    px = torch.randn(1, 3, 224, 224)
    ids = torch.randint(0, 64, (1, 6))
    all_masked = torch.full((1, 6), -100)
    _, loss = m(px, ids, all_masked)
    assert torch.isnan(loss)      # mọi target -100 -> CE trả NaN (0/0) — không dùng batch toàn mask


def test_vivlm_generate_new_tokens_only():
    m = _vlm()
    px = torch.randn(1, 3, 224, 224)
    prompt = torch.randint(0, 64, (1, 5))
    out = m.generate(px, prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape == (1, 4)


def test_trainable_phase():
    m = _vlm()
    proj_only = {id(p) for p in m.trainable_parameters("projector")}
    assert all(id(p) in proj_only for p in m.projector.parameters())
    assert not any(id(p) in proj_only for p in m.gpt.parameters())
    full = {id(p) for p in m.trainable_parameters("full")}
    assert all(id(p) in full for p in m.gpt.parameters())
    assert not any(id(p) in full for p in m.vision_encoder.parameters())
