"""
Phân loại chữ số MNIST với PyTorch — mạng nơ-ron 2 tầng ẩn.

Dataset: MNIST của torchvision (tự tải về thư mục ./data lần chạy đầu).
  - 60.000 ảnh train + 10.000 ảnh test
  - mỗi ảnh xám 28x28 -> vector 784 chiều
  - nhãn: 10 lớp (chữ số 0..9)

Kiến trúc mô hình:
  784 (input) -> 128 (ẩn 1) -> 64 (ẩn 2) -> 10 (output)

So với file digits (new_torch.py), bài này thêm 2 ý quan trọng của PyTorch:
  - DataLoader: chia dữ liệu thành các "batch" nhỏ, không nạp hết một lúc.
  - Train theo batch: cập nhật tham số nhiều lần mỗi epoch -> học nhanh hơn.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Chuẩn bị dữ liệu ---
# ToTensor: ảnh PIL [0..255] -> tensor [0..1], shape (1, 28, 28)
# Normalize: chuẩn hóa về quanh 0 bằng mean/std đã biết của MNIST -> train ổn định hơn.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# DataLoader chia dataset thành các batch; shuffle xáo trộn thứ tự mỗi epoch.
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

# --- 2. Định nghĩa mô hình: 2 tầng ẩn ---
# Flatten: ép ảnh (1, 28, 28) thành vector 784 chiều để đưa vào Linear.
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),   # tầng ẩn 1
    nn.ReLU(),
    nn.Linear(128, 64),        # tầng ẩn 2
    nn.ReLU(),
    nn.Linear(64, 10),         # tầng output: 10 logits
)

# --- 3. Loss và optimizer ---
loss_fn = nn.CrossEntropyLoss()              # softmax + NLL, nhận logits trực tiếp
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. Vòng lặp huấn luyện ---
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:        # mỗi vòng lặp = 1 batch
        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()   # xóa gradient cũ
        loss.backward()         # autograd tính gradient
        optimizer.step()        # cập nhật tham số

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss trung bình: {avg_loss:.4f}")

# --- 5. Đánh giá trên tập test ---
model.eval()
correct = 0
total = 0
with torch.no_grad():                          # tắt autograd khi suy luận
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nĐộ chính xác trên tập test: {100 * correct / total:.2f}%")
