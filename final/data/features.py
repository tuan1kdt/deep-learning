"""Precompute feature ảnh MỘT LẦN: ResNet-50 frozen → 49 vùng × 2048-d, fp16.

Đây là quyết định hạ tầng quan trọng nhất của đồ án: sau bước này mọi thí
nghiệm chỉ train decoder trên tensor có sẵn — epoch tính bằng giây/phút,
smoke chạy được trên Mac, Colab rớt session không mất gì (file nằm trên đĩa).
Đánh đổi: không augmentation được (ghi vào mục hạn chế của báo cáo).

Chạy: .venv/bin/python -m final.data.features
"""
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from final.config import Config, pick_device
from final.data.download import SPLITS, load_flickr8k

# Chuẩn hóa ImageNet — bắt buộc khớp với phân phối ResNet được pretrain
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def build_backbone() -> nn.Module:
    """ResNet-50 bỏ avgpool+fc → giữ bản đồ không gian 7×7×2048.
    Encoder chỉ chạy inference một lần nên ghim eval vĩnh viễn — không có
    chuyện BatchNorm trôi như đã phải xử lý ở midterm."""
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(backbone.children())[:-2])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def pool_to_regions(fmap: torch.Tensor) -> torch.Tensor:
    """(B,2048,7,7) → (B,49,2048); vùng i = hàng (i//7), cột (i%7) —
    quy ước này là hợp đồng với visualize.py khi vẽ heatmap đè lên ảnh."""
    return fmap.flatten(2).transpose(1, 2)


def main() -> None:
    cfg = Config()
    device = pick_device()
    ds = load_flickr8k(cfg)
    model = build_backbone().to(device)
    tf = make_transform()

    for split in SPLITS:
        out_path = cfg.features_path(split)
        if out_path.exists():
            print(f"{split}: đã có {out_path}, bỏ qua")
            continue
        rows = ds[split]
        feats = torch.empty(len(rows), cfg.num_regions, cfg.feat_dim,
                            dtype=torch.float16)
        batch, idx0 = [], 0
        with torch.no_grad():
            for i in range(len(rows)):
                batch.append(tf(rows[i]["image"].convert("RGB")))
                if len(batch) == 64 or i == len(rows) - 1:
                    x = torch.stack(batch).to(device)
                    f = pool_to_regions(model(x)).to(torch.float16).cpu()
                    feats[idx0:idx0 + len(batch)] = f
                    idx0 += len(batch)
                    batch = []
                    if idx0 % 1024 < 64:
                        print(f"  {split}: {idx0}/{len(rows)}")
        torch.save({"features": feats}, out_path)
        print(f"{split}: {tuple(feats.shape)} fp16 → {out_path}")


if __name__ == "__main__":
    main()
