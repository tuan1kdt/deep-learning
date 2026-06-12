# Đồ án giữa kỳ — Medical VQA trên VQA-RAD

Hỏi-đáp trên ảnh y khoa: cho một ảnh (X-quang / CT / MRI) và một câu hỏi tiếng
Anh, model dự đoán đáp án. Bài toán được mô hình hóa thành **classification**
trên answer vocab cố định (không phải generation).

Thiết kế chi tiết: `docs/superpowers/specs/2026-06-11-medvqa-design.md`.

## Dataset

[VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) — 1.793 QA
pairs train / 451 test. Vài con số quan trọng cần biết trước khi đọc kết quả:

| Con số | Giá trị | Ý nghĩa |
|---|---|---|
| Ảnh duy nhất trong train | 313 | ~5,7 câu hỏi cho mỗi ảnh |
| Ảnh test trùng với train | 202/203 | Split chính thức chia theo **câu hỏi**, không theo ảnh |
| Answer vocab (từ train) | 429 class | Mỗi đáp án duy nhất sau chuẩn hóa = 1 class, không có `<unk>` |
| Độ phủ vocab trên test | 334/451 = 74,1% | **Trần accuracy khả dĩ** — đáp án test ngoài vocab chắc chắn sai |
| Câu hỏi yes/no (closed) | ~52% | Phải báo cáo closed/open riêng, overall dễ gây ảo tưởng |

## Kiến trúc

```
Ảnh y khoa ─→ ResNet-50 (pretrained, freeze) ─→ v_img (768-d) ───────┐
                                  └─→ img_map 49 vùng × 768-d ───────┤
                                                                     ├─→ [FUSION] ─→ MLP head ─→ logits (429)
Câu hỏi    ─→ BERT-base (pretrained, freeze) ─→ v_txt (768-d) ───────┘
```

| Thành phần | File | Ghi chú |
|---|---|---|
| Image encoder | `models/image_encoder.py` | ResNet-50 ImageNet, freeze, xuất vector toàn cục + 49 vùng spatial |
| Text encoder | `models/text_encoder.py` | BERT-base, freeze, mean-pooling (mặc định) hoặc `[CLS]` |
| Fusion ×3 | `models/fusion.py` | `concat` / `hadamard` / `cross_attention` — trục ablation chính |
| Model + head | `models/vqa_model.py` | MLP 768→1024→429, dropout 0.5; ~5,6–6,7M tham số trainable |
| Config trung tâm | `config.py` | Mọi lựa chọn thí nghiệm đi qua đây |

## Cách chạy

```bash
source .venv/bin/activate

# 1. Chuẩn bị data (idempotent, ~34MB)
python -m midterm.data.download
python -m midterm.data.vocab

# 2. Ba thí nghiệm của báo cáo — chỉ khác fusion, cùng seed
python -m midterm.train --fusion concat
python -m midterm.train --fusion hadamard
python -m midterm.train --fusion cross_attention

# 3. Đánh giá trên test (451 mẫu, chỉ chạy ở bước cuối)
python -m midterm.evaluate --checkpoint midterm/checkpoints/concat.pt

# 4. Demo: ảnh + câu hỏi → top-5 đáp án (+ attention overlay nếu cross_attention)
python -m midterm.demo --checkpoint midterm/checkpoints/cross_attention.pt \
    --image x.jpg --question "is there cardiomegaly?"
```

Smoke test local (Mac MPS, ~2 phút — subset 128 mẫu, 2 epoch):

```bash
python -m midterm.train --fusion concat --smoke
```

Train thật trên Colab GPU: mở `colab_train.ipynb` (notebook mỏng, chỉ gọi CLI).

Mỗi run lưu `outputs/<run_name>/` (config, history, curves.png) và checkpoint
tốt nhất theo val accuracy tại `checkpoints/<run_name>.pt`.

## Kết quả

> Điền sau khi train trên Colab (số liệu lấy từ `python -m midterm.evaluate`).

| Fusion | Overall | Closed (yes/no) | Open |
|---|---|---|---|
| concat | — | — | — |
| hadamard | — | — | — |
| cross_attention | — | — | — |

## Ghi chú thiết kế (cho vấn đáp)

**Vì sao BatchNorm bị ghim ở eval mode?** `requires_grad=False` chỉ chặn
optimizer cập nhật weight. Nếu để `model.train()` lan vào backbone, BatchNorm
vẫn cập nhật running mean/var theo ảnh y khoa — encoder "frozen" âm thầm đổi
hành vi và kết quả không tái lập. Nên `ImageEncoder.train()` được override để
backbone luôn ở eval (gradient vẫn chảy bình thường khi unfreeze layer4).

**Vì sao mean-pooling thay vì `[CLS]`?** `[CLS]` của BERT được pretrain cho
next-sentence prediction — khi không fine-tune, nó là biểu diễn câu yếu.
Mean-pooling hidden states (có mask) thường tốt hơn rõ rệt với BERT freeze.
`--text-pool cls` giữ làm đối chứng.

**Val split theo QA pair có phải data leakage không?** Không — đây là lựa chọn
có chủ đích. 202/203 ảnh test cũng xuất hiện trong train: split chính thức của
VQA-RAD vốn chia theo câu hỏi. Val theo QA pair (10%, seed 42) vì vậy khớp đúng
"điều kiện thi" của test: ảnh đã thấy, câu hỏi mới. Split theo ảnh sẽ *khó hơn*
điều kiện test thật.

**Confound trong ablation:** `cross_attention` nhận 49 vùng spatial mà
`concat`/`hadamard` không dùng — khác biệt kết quả gộp cả "cơ chế fusion" lẫn
"có thông tin spatial". Báo cáo phải thừa nhận điều này khi kết luận.

**Vì sao không horizontal flip?** Ảnh y khoa có tính trái/phải (tim bên trái,
gan bên phải) — flip tạo ra ảnh sai về mặt giải phẫu.

**Chống overfit trên 1.6k mẫu:** freeze encoder (chỉ ~6M tham số trainable),
dropout 0.5, random resized crop nhẹ (scale 0.9–1.0), early stopping theo val
accuracy (patience 5).
