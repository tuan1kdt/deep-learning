"""
Phân loại chữ số viết tay với PyTorch — mạng nơ-ron 2 tầng ẩn.

Dataset: load_digits của scikit-learn (có sẵn, không cần tải mạng).
  - 1797 ảnh chữ số 8x8 (giá trị pixel 0..16)
  - mỗi ảnh -> vector 64 chiều
  - nhãn: 10 lớp (chữ số 0..9)

Kiến trúc mô hình:
  64 (input) -> 32 (ẩn 1) -> 16 (ẩn 2) -> 10 (output)

Đây là bước tiếp theo sau khi đã hiểu backprop thủ công ở Day 1-2:
PyTorch tự lo phần gradient (autograd), ta chỉ cần định nghĩa mô hình + loss.
"""

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- 1. Chuẩn bị dữ liệu ---
digits = load_digits()
X = digits.data / 16.0          # chuẩn hóa pixel về [0, 1] giúp train ổn định hơn
y = digits.target

# Tách train/test để kiểm tra mô hình có học vẹt (overfit) hay không
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Chuyển sang tensor — float cho đầu vào, long cho nhãn (yêu cầu của CrossEntropyLoss)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# --- 2. Định nghĩa mô hình: 2 tầng ẩn ---
# nn.Sequential xếp các tầng nối tiếp nhau.
# ReLU là hàm kích hoạt phi tuyến — không có nó, nhiều tầng tuyến tính
# gộp lại vẫn chỉ là một phép tuyến tính.
model = nn.Sequential(
    nn.Linear(64, 32),   # tầng ẩn 1
    nn.ReLU(),
    nn.Linear(32, 16),   # tầng ẩn 2
    nn.ReLU(),
    nn.Linear(16, 10),   # tầng output: 10 điểm số (logits) cho 10 lớp
)

# --- 3. Loss và optimizer ---
# CrossEntropyLoss = softmax + negative log-likelihood, dùng cho phân loại nhiều lớp.
# Nó nhận logits trực tiếp (không cần tự thêm softmax).
loss_fn = nn.CrossEntropyLoss()
# Adam: thuật toán tối ưu tự điều chỉnh learning rate, hội tụ nhanh hơn SGD thường.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 4. Vòng lặp huấn luyện ---
EPOCHS = 100
for epoch in range(EPOCHS):
    # Forward: dự đoán + tính loss
    logits = model(X_train)
    loss = loss_fn(logits, y_train)

    # Backward: autograd tính toàn bộ gradient cho ta
    optimizer.zero_grad()   # xóa gradient cũ (PyTorch cộng dồn gradient theo mặc định)
    loss.backward()         # tính dLoss/dW cho mọi tham số
    optimizer.step()        # cập nhật tham số theo gradient

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} - loss: {loss.item():.4f}")

# --- 5. Đánh giá trên tập test ---
# torch.no_grad(): tắt autograd khi suy luận -> nhanh hơn, đỡ tốn bộ nhớ.
with torch.no_grad():
    test_logits = model(X_test)
    preds = test_logits.argmax(dim=1)        # lớp có điểm cao nhất
    accuracy = (preds == y_test).float().mean()

print(f"\nĐộ chính xác trên tập test: {accuracy.item() * 100:.2f}%")
