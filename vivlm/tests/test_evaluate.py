import numpy as np
import torch

from vivlm.evaluate import beam_generate, perplexity, score_captions
from vivlm.models.gpt import GPT
from vivlm.models.projector import PixelShuffleProjector
from vivlm.models.vivlm import ViVLM
from vivlm.tests.test_fusion import TINY, StubEncoder


def test_score_captions_perfect_match():
    gts = {0: ["ba chiếc thuyền trên sông", "thuyền trên sông"],
           1: ["cậu bé đá bóng"]}
    res = {0: "ba chiếc thuyền trên sông", 1: "cậu bé đá bóng"}
    s = score_captions(gts, res)
    assert s["Bleu_4"] > 0.9 and s["CIDEr"] > 1.0


def test_score_captions_case_whitespace_insensitive():
    gts = {0: ["Ba chiếc Thuyền"]}
    a = score_captions(gts, {0: "ba  chiếc thuyền"})
    b = score_captions(gts, {0: "BA CHIẾC THUYỀN"})
    assert abs(a["CIDEr"] - b["CIDEr"]) < 1e-6


def test_perplexity_random_model(tmp_path):
    p = tmp_path / "v.bin"
    np.random.randint(0, TINY.vocab_size, 5000,
                      dtype=np.uint16).tofile(p)
    m = GPT(TINY).eval()
    ppl = perplexity(m, str(p), TINY.context, "cpu", iters=5)
    assert 10 < ppl < 5000        # model ngẫu nhiên ~ vocab_size=64 -> quanh 64


def test_bits_per_char(tmp_path, tiny_tokenizer):
    from vivlm.evaluate import bits_per_char
    ids = tiny_tokenizer.encode("ba chiếc thuyền trên sông " * 300).ids
    np.asarray(ids, dtype=np.uint16).tofile(tmp_path / "v.bin")
    cfg = TINY.__class__(**{**TINY.__dict__, "vocab_size": 300})
    m = GPT(cfg).eval()
    bpc = bits_per_char(m, tiny_tokenizer, str(tmp_path / "v.bin"),
                        cfg.context, "cpu", iters=3)
    assert 0.5 < bpc < 20        # model ngẫu nhiên: vài bit/ký tự


def test_beam1_equals_greedy():
    torch.manual_seed(0)
    vlm = ViVLM(GPT(TINY), StubEncoder(),
                PixelShuffleProjector(768, TINY.d_model)).eval()
    px = torch.randn(1, 3, 224, 224)
    prompt = torch.randint(0, 64, (1, 5))
    greedy = vlm.generate(px, prompt, 6, temperature=0.0)
    beam = beam_generate(vlm, px, prompt, beam_size=1, max_new=6, eos_id=-1)
    assert torch.equal(greedy, beam)
