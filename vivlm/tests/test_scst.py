import torch

from vivlm.models.gpt import GPT
from vivlm.models.projector import PixelShuffleProjector
from vivlm.models.vivlm import ViVLM
from vivlm.scst import caption_mask, scst_loss, sequence_logprobs
from vivlm.tests.test_fusion import TINY, StubEncoder


def test_caption_mask_first_eos_inclusive():
    sampled = torch.tensor([[5, 3, 0, 7, 0],       # eos=0 ở vị trí 2
                            [1, 2, 3, 4, 5]])      # không eos
    m = caption_mask(sampled, eos_id=0)
    assert m.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_scst_loss_zero_advantage():
    lp = torch.randn(3, 5)
    mask = torch.ones(3, 5)
    assert scst_loss(lp, mask, torch.zeros(3)).item() == 0.0


def test_scst_loss_value():
    lp = torch.full((1, 2), -1.0)
    loss = scst_loss(lp, torch.ones(1, 2), torch.tensor([0.5]))
    assert abs(loss.item() - 1.0) < 1e-6           # -(0.5 * -2) = 1.0


def test_sequence_logprobs_shape_and_grad():
    vlm = ViVLM(GPT(TINY), StubEncoder(),
                PixelShuffleProjector(768, TINY.d_model))
    px = torch.randn(2, 3, 224, 224)
    prompt = torch.randint(1, 64, (2, 4))
    sampled = torch.randint(1, 64, (2, 6))
    lp = sequence_logprobs(vlm, px, prompt, sampled)
    assert lp.shape == (2, 6)
    assert (lp <= 0).all()                          # log prob
    lp.sum().backward()                             # có grad về GPT
    assert vlm.gpt.tok_emb.weight.grad is not None
