"""
Day 7 — Multimodal Fusion: kết hợp ảnh (CNN) và text (LSTM) trong một model.

Bài toán: cho ảnh MNIST chứa chữ số d VÀ một caption tiếng Việt "cộng k"
(k viết bằng chữ, vd "cộng ba"), dự đoán (d + k) mod 10 — phân loại 10 lớp.

Điểm then chốt của thiết kế: CHỈ ảnh hoặc CHỈ text đều không đủ thông tin
để đoán nhãn (mỗi nguồn riêng lẻ chỉ đạt ~10% accuracy — ngang đoán mò).
Model buộc phải HỢP NHẤT (fusion) hai nguồn: đây chính là "Kiến trúc 1"
trên bảng — ảnh → CNN, text → LSTM, hai vector đặc trưng concat lại,
đi qua combined model (FC) rồi tính loss. Train end-to-end: gradient từ
loss lan ngược qua FC → concat → cả CNN lẫn LSTM, một optimizer duy nhất
cập nhật toàn bộ tham số ("update parameter ở đây").

Chạy:  python day7/01_multimodal_fusion.py
Xem thêm lý thuyết trong day7/README.md.
"""

import os
import random

import matplotlib

matplotlib.use("Agg")  # backend không cần GUI — vẽ thẳng ra file .png
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Cấu hình chung
# ---------------------------------------------------------------------------
SEED = 42
DAY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DAY_DIR, "..", "data")  # MNIST đã có sẵn ở data/MNIST

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
D_IMG = 64      # số chiều vector đặc trưng nhánh ảnh (đầu ra CNN)
D_EMB = 16      # số chiều embedding của một token text
D_TXT = 32      # số chiều hidden LSTM = vector đặc trưng nhánh text
N_CLASSES = 10  # nhãn = (d + k) mod 10 nên vẫn là 10 lớp

random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Vocab tí hon cho caption "cộng <số>"
# ---------------------------------------------------------------------------
# Caption luôn có đúng 2 token nên không cần padding thật sự, nhưng vẫn dành
# chỗ cho PAD=0 theo convention chung — code mở rộng sang câu dài dễ hơn.
DIGIT_WORDS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
PAD = 0
VOCAB = {"<pad>": PAD, "cộng": 1}
for _w in DIGIT_WORDS:
    VOCAB[_w] = len(VOCAB)
VOCAB_SIZE = len(VOCAB)  # 12 token


# ---------------------------------------------------------------------------
# Dataset: ghép mỗi ảnh MNIST với một caption "cộng k" sinh tự động
# ---------------------------------------------------------------------------
class MultimodalMNIST(Dataset):
    """Mỗi sample = (ảnh chữ số d, caption "cộng k", nhãn (d+k)%10).

    k được rút ngẫu nhiên MỘT LẦN lúc khởi tạo (seed cố định) rồi giữ nguyên,
    để mỗi lần chạy/eval đều thấy đúng cùng một bộ dữ liệu — kết quả tái lập.
    """

    def __init__(self, train: bool):
        self.base = datasets.MNIST(
            root=DATA_DIR,
            train=train,
            download=True,  # đã có sẵn trên đĩa thì torchvision bỏ qua
            transform=transforms.ToTensor(),  # ảnh PIL -> FloatTensor [1,28,28], giá trị [0,1]
        )
        # seed tách biệt cho train/test để hai tập không "trùng nhịp" k
        rng = random.Random(SEED if train else SEED + 1)
        self.ks = [rng.randrange(10) for _ in range(len(self.base))]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, digit = self.base[i]
        k = self.ks[i]
        # caption "cộng k" -> chuỗi 2 chỉ số token cho Embedding/LSTM
        caption = torch.tensor([VOCAB["cộng"], VOCAB[DIGIT_WORDS[k]]], dtype=torch.long)
        label = torch.tensor((digit + k) % 10, dtype=torch.long)
        return img, caption, label


def make_loaders():
    train_ds = MultimodalMNIST(train=True)
    test_ds = MultimodalMNIST(train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, test_loader


# --- kiểm tra nhanh data pipeline (Task 2 sẽ thay phần này bằng model) ---
if __name__ == "__main__":
    train_loader, test_loader = make_loaders()
    img, caption, label = next(iter(train_loader))
    print(f"batch ảnh   : {tuple(img.shape)}")      # (128, 1, 28, 28)
    print(f"batch text  : {tuple(caption.shape)}")  # (128, 2)
    print(f"batch nhãn  : {tuple(label.shape)}")    # (128,)
    inv = {v: k for k, v in VOCAB.items()}
    ds = train_loader.dataset
    for i in range(3):
        im, cap, lab = ds[i]
        digit = ds.base[i][1]
        words = " ".join(inv[t.item()] for t in cap)
        print(f'sample {i}: ảnh số {digit}, caption "{words}" -> nhãn {lab.item()}')
