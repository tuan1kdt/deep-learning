# MedVQA Desktop Demo (Go + Wails)

App demo model MedVQA: upload ảnh y khoa + nhập câu hỏi tiếng Anh → đáp án
(top-5) và heatmap attention (khi checkpoint là `cross_attention`).

## Kiến trúc

```
React (frontend) ──Wails bindings──► Go (app.go) ──HTTP localhost──► Python sidecar
                                                                     (midterm/serve.py)
```

Sidecar (`midterm/serve.py`, Flask) load model PyTorch một lần và giữ nóng; Go
spawn nó lúc khởi động (cổng tự do) và kill khi đóng app. Tái dùng toàn bộ logic
model trong `midterm/` — không nhân đôi.

## Yêu cầu

- `.venv` ở repo root đã cài `requirements.txt` (gồm `flask`).
- Có ít nhất 1 checkpoint trong `midterm/checkpoints/` (vd `concat.pt`).
  Để có heatmap cần `cross_attention.pt`.
- Go 1.26+, Wails v2.12+, Node 18+.

## Chạy (dev)

```bash
cd midterm/desktop
wails dev
```

App tự tìm repo root (thư mục chứa `.venv` + `midterm`). Nếu chạy từ nơi khác,
đặt biến môi trường `DEEPLEARNING_ROOT=/đường/dẫn/repo`.

## Build (chạy local)

```bash
cd midterm/desktop
wails build
# → build/bin/desktop.app  (chạy trên máy có sẵn .venv tại repo root)
```

## Cách dùng

1. Chọn checkpoint ở góc phải (đổi sẽ reload model ~20s).
2. Kéo-thả / chọn ảnh.
3. Gõ câu hỏi tiếng Anh → **Run**.
4. Xem đáp án top-5; với `cross_attention` xem thêm heatmap.
