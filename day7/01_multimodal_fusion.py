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


# ---------------------------------------------------------------------------
# Nhánh ảnh: CNN nhỏ — 2 tầng conv, mỗi tầng giảm nửa kích thước không gian
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B,16,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,16,14,14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [B,32,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,32,7,7]
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, D_IMG),                # nén về vector D_IMG chiều
            nn.ReLU(),
        )

    def forward(self, img):
        return self.net(img)  # [B, D_IMG]


# ---------------------------------------------------------------------------
# Nhánh text: Embedding + LSTM — lấy hidden state cuối làm đặc trưng cả câu
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, D_EMB, padding_idx=PAD)
        self.lstm = nn.LSTM(D_EMB, D_TXT, batch_first=True)

    def forward(self, caption):
        # caption [B,2] -> emb [B,2,D_EMB] -> LSTM đọc lần lượt từng token
        _, (h, _) = self.lstm(self.emb(caption))
        # h [1,B,D_TXT]: hidden state SAU token cuối — tóm tắt toàn bộ câu
        return h[-1]  # [B, D_TXT]


# ---------------------------------------------------------------------------
# Fusion + combined model: concat 2 vector đặc trưng rồi qua FC head
# ---------------------------------------------------------------------------
class MultimodalNet(nn.Module):
    """drop_img / drop_txt: thay đặc trưng nhánh đó bằng 0 — dùng cho ablation
    sau khi train, chứng minh model thật sự cần cả hai nguồn thông tin."""

    def __init__(self):
        super().__init__()
        self.img_enc = ImageEncoder()
        self.txt_enc = TextEncoder()
        self.head = nn.Sequential(
            nn.Linear(D_IMG + D_TXT, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, img, caption, drop_img=False, drop_txt=False):
        f_img = self.img_enc(img)      # [B, D_IMG]
        f_txt = self.txt_enc(caption)  # [B, D_TXT]
        if drop_img:
            f_img = torch.zeros_like(f_img)
        if drop_txt:
            f_txt = torch.zeros_like(f_txt)
        # ĐÂY là "fusion" trên bảng: ghép 2 modality thành 1 vector duy nhất.
        # Concat là cách đơn giản nhất; gradient từ head sẽ tự tách về đúng
        # từng nhánh (xem README — mục luồng gradient).
        fused = torch.cat([f_img, f_txt], dim=1)  # [B, D_IMG + D_TXT]
        return self.head(fused)  # logits [B, N_CLASSES]


# --- kiểm tra nhanh forward pass (Task 3 sẽ thay bằng vòng train đầy đủ) ---
if __name__ == "__main__":
    train_loader, test_loader = make_loaders()
    model = MultimodalNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model có {n_params:,} tham số")
    img, caption, label = next(iter(train_loader))
    logits = model(img.to(DEVICE), caption.to(DEVICE))
    print(f"logits: {tuple(logits.shape)}")  # (128, 10)
    loss = nn.CrossEntropyLoss()(logits, label.to(DEVICE))
    print(f"loss khởi đầu: {loss.item():.3f} (kỳ vọng ~ln(10) ≈ 2.303)")
