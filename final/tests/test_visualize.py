import torch
from PIL import Image

from final.visualize import attention_figure


def test_attention_figure_tao_dung_so_o():
    img = Image.new("RGB", (224, 224), color=(100, 150, 60))
    words = ["a", "dog", "runs"]
    attn = torch.rand(3, 49)
    fig = attention_figure(img, words, attn)
    # 1 ô ảnh gốc + 3 ô từ
    assert len(fig.axes) == 4
    import matplotlib.pyplot as plt
    plt.close(fig)
