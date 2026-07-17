# Đồ án cuối kỳ — Image Captioning trên Flickr8k

Sinh mô tả ảnh: đi nhánh **generation** trong sơ đồ kiến trúc multimodal của
giảng viên (đồ án giữa kỳ đã đi nhánh classification — hai đồ án ghép lại phủ
trọn sơ đồ). Thiết kế chi tiết:
`docs/superpowers/specs/2026-07-11-final-image-captioning-design.md`.

## Kiến trúc

```
ảnh ─→ ResNet-50 (pretrained, freeze, precompute) ─→ 49 vùng × 2048 ─→ proj 512 ─┐
                                                                                  ├─ cross-attention (fusion) ─→ sinh từng từ
câu đang sinh ─→ embedding (+ self-attention / hồi quy LSTM) ────────────────────┘
```

Hai decoder so sánh (trục thí nghiệm chính):
1. **LSTM + Bahdanau attention** (Show, Attend and Tell) — bản PyTorch của cơ chế đã tự cài NumPy ở day6.
2. **Transformer decoder** 3 tầng, tự cài block để xuất cross-attention.

## Dữ liệu

[jxie/flickr8k](https://huggingface.co/datasets/jxie/flickr8k): 8.000 ảnh
(6.000 train / 1.000 val / 1.000 test — Karpathy split), 5 caption/ảnh.
Vocab mức từ min_freq=5, max 20 từ + bos/eos.

## Cách chạy

Toàn bộ pipeline (download → vocab → features → 3 run train → evaluate → visualize)
nằm trong `final/run_all.sh`, chạy được cả trên Mac (smoke) lẫn trên máy remote
trainbox (train thật). Mỗi bước train được guard: nếu checkpoint tương ứng
(`final/checkpoints/<run>.pt`) đã tồn tại thì bỏ qua, nên có thể chạy lại script
an toàn sau khi rớt giữa chừng.

### (a) Mac local — smoke test

Kiểm tra pipeline chạy được từ đầu đến cuối, mỗi run train chỉ 512 mẫu + 2 epoch:

```bash
source .venv/bin/activate
./final/run_all.sh --smoke
```

### (b) Máy remote trainbox — train thật

Trainbox: Windows + WSL2 Ubuntu, GPU RTX PRO 4000 Blackwell 24GB. Venv riêng tại
`~/work/.venv` (Python 3.12, torch cu128 — Blackwell bắt buộc cu128 trở lên, cu121
không nhận GPU này).

Chạy trong `tmux` để không mất session khi mất kết nối SSH:

```bash
tmux new -s train
PY=~/work/.venv/bin/python ./final/run_all.sh
# Ctrl-b d để detach, tmux attach -t train để quay lại xem tiến trình
```

Kéo checkpoints + outputs về máy dev bằng rsync:

```bash
rsync -avz trainbox:~/work/deepLearning/final/checkpoints/ final/checkpoints/
rsync -avz trainbox:~/work/deepLearning/final/outputs/ final/outputs/
```

> Ghi chú: HF token trên máy dev đã hết hạn. Nếu bước download gặp lỗi 401,
> chạy lại với `HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python -m final.data.download`.

## Kết quả

> Điền sau khi train xong trên trainbox (số từ `python -m final.evaluate`).

| Run | mode | BLEU-1 | BLEU-4 | CIDEr |
|---|---|---|---|---|
| lstm | greedy | — | — | — |
| lstm | beam5 | — | — | — |
| lstm_noattn | greedy | — | — | — |
| transformer | greedy | — | — | — |
| transformer | beam5 | — | — | — |

## Ghi chú thiết kế (cho báo cáo)

**Vì sao precompute feature?** Encoder frozen → feature bất biến → tính một
lần, train decoder nhanh hơn ~20×, rớt session không mất gì. Đánh đổi:
không augmentation được (ghi vào hạn chế).

**Vì sao tự cài Transformer block?** `nn.TransformerDecoder` không trả
cross-attention weights — mà heatmap "từ nào nhìn vùng nào" là hình quan
trọng nhất của báo cáo.

**Vì sao tie embedding với output layer?** Giảm ~1.5M tham số trên vocab 3k
và buộc không gian biểu diễn vào/ra nhất quán — quan trọng với data nhỏ.

**Vì sao CIDEr tự tokenize thay vì PTBTokenizer?** Né dependency Java trong
pycocoevalcap; cả hyps lẫn refs tokenize cùng một kiểu nên so sánh nội bộ
vẫn công bằng (lệch nhỏ so với số chuẩn COCO — đã ghi chú).

**Vì sao init embedding std=0.02 khi tie weight?** Khi tie embedding với
output layer, PyTorch mặc định khởi tạo embedding N(0,1) — quá lớn cho một
layer cũng đóng vai trò output projection, khiến logits ban đầu bùng nổ.
Phát hiện qua smoke run: transformer có cross-entropy ban đầu ~267 (đáng lẽ
~ln(vocab_size) ≈ 8) do bug này. Fix: khởi tạo N(0, 0.02) như GPT-2/BERT —
sẽ nêu lại trong báo cáo như một lỗi cụ thể đã gặp và cách debug.
