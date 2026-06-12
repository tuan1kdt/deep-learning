"""Image encoder: ResNet-50 pretrained ImageNet, mặc định freeze toàn bộ.

Xuất 2 dạng đặc trưng:
- v_img   (B, d_model): vector toàn cục (avg pool) — cho fusion concat/hadamard.
- img_map (B, 49, d_model): 7×7 = 49 vùng không gian — cho cross_attention.
Cả hai chiếu qua Linear riêng về d_model để khớp chiều với vector câu hỏi.
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ImageEncoder(nn.Module):
    def __init__(self, d_model: int = 768, unfreeze_last_block: bool = False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Bỏ avgpool + fc cuối, giữ phần conv → feature map (B, 2048, 7, 7)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            # layer4 (block conv sâu nhất) là child cuối của backbone;
            # train với LR riêng thấp hơn — xem param groups trong train.py
            for p in self.backbone[-1].parameters():
                p.requires_grad = True

        self.proj_global = nn.Linear(2048, d_model)
        self.proj_regions = nn.Linear(2048, d_model)

    def train(self, mode: bool = True):
        """BatchNorm của backbone LUÔN ở eval mode, kể cả lúc train.

        requires_grad=False chỉ chặn cập nhật weight qua optimizer; nếu để
        train mode, BatchNorm vẫn cập nhật running mean/var theo ảnh y khoa →
        encoder "frozen" âm thầm thay đổi hành vi và kết quả không tái lập.
        Gradient vẫn chảy bình thường qua module ở eval mode (cần cho
        unfreeze_last_block).
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor):
        fmap = self.backbone(images)               # (B, 2048, 7, 7)
        v_global = fmap.mean(dim=(2, 3))           # avg pool → (B, 2048)
        regions = fmap.flatten(2).transpose(1, 2)  # (B, 49, 2048)
        return self.proj_global(v_global), self.proj_regions(regions)
