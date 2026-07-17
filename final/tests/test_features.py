import pytest
import torch

from final.data.features import build_backbone, make_transform, pool_to_regions


def test_pool_to_regions_shape_va_thu_tu():
    fmap = torch.zeros(2, 2048, 7, 7)
    fmap[0, :, 0, 1] = 1.0                    # hàng 0, cột 1 → vùng index 1
    fmap[0, :, 2, 3] = 5.0                    # hàng 2, cột 3 → vùng 2*7+3=17
    out = pool_to_regions(fmap)
    assert out.shape == (2, 49, 2048)
    assert out[0, 1].mean().item() == 1.0
    assert out[0, 17].mean().item() == 5.0
    assert out[0, 0].abs().sum().item() == 0.0


def test_transform_ra_dung_kich_thuoc():
    from PIL import Image

    img = Image.new("RGB", (500, 375), color=(120, 30, 200))
    x = make_transform()(img)
    assert x.shape == (3, 224, 224)


@pytest.mark.integration  # tải trọng số ResNet-50 (~100MB, có thể đã cache)
def test_backbone_shape_thật():
    model = build_backbone()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 2048, 7, 7)
