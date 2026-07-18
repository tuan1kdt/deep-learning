import numpy as np
import torch

from vivlm.data.loader import TokenBin


def _make_bin(tmp_path, n=1000):
    p = tmp_path / "t.bin"
    np.arange(n, dtype=np.uint16).tofile(p)     # token i tại vị trí i
    return str(p)


def test_shapes_dtype(tmp_path):
    tb = TokenBin(_make_bin(tmp_path))
    x, y = tb.sample(4, 16)
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == torch.long
    assert len(tb) == 1000


def test_y_is_shifted_x(tmp_path):
    tb = TokenBin(_make_bin(tmp_path))
    x, y = tb.sample(8, 32)
    assert torch.equal(y, x + 1)                # vì data là dãy tăng dần


def test_generator_reproducible(tmp_path):
    tb = TokenBin(_make_bin(tmp_path))
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    x1, _ = tb.sample(4, 16, generator=g1)
    x2, _ = tb.sample(4, 16, generator=g2)
    assert torch.equal(x1, x2)
