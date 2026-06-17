# MedVQA — Thiết kế hệ thống Hỏi-đáp trên ảnh y khoa (VQA-RAD)

**Ngày:** 2026-06-11
**Trạng thái:** Đã duyệt qua brainstorming. Cập nhật 2026-06-12 sau review: BN eval mode,
mean-pooling text, val split có chủ đích, bỏ `<unk>`, ghi nhận confound ablation.
**Phạm vi:** Đồ án giữa kỳ — thư mục `midterm/`

## 1. Mục tiêu & tiêu chí thành công

Xây dựng model Medical Visual Question Answering trên dataset VQA-RAD: cho một ảnh
y khoa (X-quang, CT, MRI) + một câu hỏi tiếng Anh, model dự đoán đáp án.

Tiêu chí thành công (theo thứ tự ưu tiên người dùng đã chốt):

1. **Code sạch + hiểu sâu** — code là tài liệu học tập, đọc từ trên xuống như
   tutorial, comment tiếng Việt giải thích *why*, giống phong cách `day1/`–`day3/`.
   Người làm phải giải thích được mọi thành phần khi vấn đáp.
2. **Báo cáo + thí nghiệm so sánh** — trục ablation chính là **fusion strategy**
   (3 biến thể, cùng codebase, chỉ đổi config).
3. **Demo chạy được** — script inference: ảnh + câu hỏi vào, đáp án ra.

Accuracy cao không phải ưu tiên hàng đầu, nhưng cần đủ tốt để báo cáo thuyết
phục (kỳ vọng overall ~50–65% trên test với encoder pretrained bị freeze).

## 2. Ràng buộc đã chốt

- **Được dùng pretrained encoder** (ResNet ImageNet, BERT) — chỉ tự xây fusion + head.
- **Train trên Google Colab / Kaggle GPU** (T4/P100); smoke test local trên Mac (MPS).
- **Bài toán dạng classification** trên answer vocab cố định (không phải generation).
- Dataset: VQA-RAD từ HuggingFace (`flaviagiammarino/vqa-rad`), 1.793 train / 451 test,
  features `image`, `question`, `answer`. Đã tải sẵn tại `midterm/data/vqa_rad/`.
- Python 3.14, venv `.venv/` với torch 2.12, torchvision 0.27, transformers 5.10, datasets 5.0.
- Code cũ trong `midterm/` chỉ còn `.pyc` — xây lại từ đầu, giữ nguyên bố cục
  module cũ (`config`, `data/download`, `data/vocab`, `data/dataset`).

## 3. Kiến trúc model

```
Ảnh y khoa ─→ ResNet-50 (pretrained, freeze) ─→ v_img (2048-d) ──┐
                                  └─→ feature map 7×7×2048 ──────┤
                                                                 ├─→ [FUSION] ─→ MLP head ─→ logits
Câu hỏi    ─→ BERT-base (pretrained, freeze) ─→ v_txt (768-d) ───┘
```

### 3.1 Image encoder (`models/image_encoder.py`)

- ResNet-50 pretrained ImageNet từ torchvision, bỏ lớp FC cuối.
- Xuất 2 dạng đặc trưng:
  - **Vector toàn cục** 2048-d (sau adaptive average pool) — cho fusion `concat`/`hadamard`.
  - **Feature map không gian** 7×7×2048, reshape thành 49 vector "vùng ảnh" — cho `cross_attention`.
- Cả hai chiếu qua `Linear` về `d_model = 768`.
- Mặc định **freeze toàn bộ**; cờ config `unfreeze_last_block` mở block conv cuối
  (layer4) làm thí nghiệm phụ.
- **BatchNorm luôn chạy ở eval mode** (running stats đóng băng), kể cả lúc train
  và kể cả khi `unfreeze_last_block`. Lý do: chỉ đặt `requires_grad=False` là chưa
  đủ — nếu gọi `model.train()` toàn cục, BN vẫn cập nhật running stats theo ảnh
  y khoa, khiến encoder "frozen" âm thầm thay đổi hành vi và kết quả không tái lập.

### 3.2 Text encoder (`models/text_encoder.py`)

- `bert-base-uncased` từ transformers (câu hỏi VQA-RAD là tiếng Anh).
- Vector câu hỏi 768-d lấy theo config `text_pool`:
  - `mean` (**mặc định**) — mean-pooling hidden states lớp cuối, có attention mask.
    `[CLS]` của BERT freeze được pretrain cho next-sentence prediction nên là biểu
    diễn câu yếu; mean-pooling thường tốt hơn rõ rệt khi không fine-tune.
  - `cls` — embedding `[CLS]`, giữ làm đối chứng (thí nghiệm phụ rẻ tiền).
  Vector này dùng cho cả 3 fusion; với `cross_attention` nó là query duy nhất.
- Mặc định **freeze toàn bộ**. Tên model là config — đổi sang
  `emilyalsentzer/Bio_ClinicalBERT` là một thí nghiệm phụ tùy thời gian.

### 3.3 Fusion (`models/fusion.py`) — trục ablation chính

Cả 3 module cùng interface: nhận `(v_img, img_map, v_txt)` → trả vector 768-d.

| Tên config | Công thức | Ý nghĩa |
|---|---|---|
| `concat` | `Linear([v_img ; v_txt]) → ReLU` | Baseline đơn giản nhất |
| `hadamard` | `v_img ⊙ v_txt` (sau khi chiếu cùng chiều) | Tương tác nhân giữa hai modality |
| `cross_attention` | Multi-head attention: Q = v_txt, K/V = 49 vùng ảnh; cộng residual + LayerNorm | Câu hỏi "nhìn" vào vùng ảnh liên quan; attention weights trực quan hóa được |

> **Lưu ý diễn giải ablation:** `cross_attention` nhận 49 vùng spatial mà
> `concat`/`hadamard` không có, nên khác biệt kết quả gộp cả hai yếu tố "cơ chế
> fusion" và "có thông tin spatial". Báo cáo phải thừa nhận confound này khi kết
> luận; không bắt buộc thêm variant để gỡ (ngoài phạm vi).

### 3.4 Classifier head (trong `models/vqa_model.py`)

- MLP: `Linear(768→1024) → ReLU → Dropout(0.5) → Linear(1024→|vocab|)`.
- Loss: `CrossEntropyLoss`.
- `VQAModel` ghép 4 thành phần theo config; tổng tham số trainable ~vài triệu
  (encoder freeze) → train được trên T4 trong vài phút/epoch.

## 4. Data pipeline

### 4.1 Download (`data/download.py`)

- Tải `flaviagiammarino/vqa-rad` từ HF Hub → `save_to_disk("midterm/data/vqa_rad")`.
- Idempotent: nếu đã có trên đĩa thì bỏ qua. Trên Colab chạy lại lệnh này (~34MB).

### 4.2 Answer vocab (`data/vocab.py`)

- Build từ **train split duy nhất**. Chuẩn hóa đáp án: lowercase, strip khoảng
  trắng và dấu câu thừa, gộp khoảng trắng liên tiếp. Mỗi đáp án duy nhất sau
  chuẩn hóa = 1 class — đúng **429 class, không có token `<unk>`** (mọi đáp án
  train đều nằm trong vocab theo cách build, nên `<unk>` sẽ là class chết không
  bao giờ làm target). Độ phủ vocab trên test: **334/451 = 74,1%** — trần
  accuracy khả dĩ, nhất quán với kỳ vọng 50–65%.
- Lưu `midterm/data/answer_vocab.json` (mapping answer ↔ index). File hiện có
  do code cũ tạo chứa `<unk>` (433 entries) — script mới phải **build lại, ghi đè**.
- Đáp án test ngoài vocab → dự đoán chắc chắn sai; script in **độ phủ vocab trên
  test** để báo cáo minh bạch.

### 4.3 Dataset (`data/dataset.py`)

- `VQARadDataset(torch.utils.data.Dataset)` đọc từ disk bằng `load_from_disk`.
- Ảnh: convert RGB, resize 224×224, chuẩn hóa mean/std ImageNet.
  Augmentation nhẹ chỉ cho train và tắt được qua config: random resized crop
  (scale 0.9–1.0). **Không horizontal flip** — ảnh y khoa có tính trái/phải.
- Câu hỏi: tokenize bằng BERT tokenizer, pad/truncate `max_len = 32`.
- **Validation split**: tách 10% từ train với seed 42, **theo QA pair** (VQA-RAD
  không có val chính thức). Đây là lựa chọn có chủ đích, không phải sơ suất:
  train có 1.793 QA pairs nhưng chỉ **313 ảnh duy nhất** (~5,7 câu hỏi/ảnh), và
  **202/203 ảnh test cũng xuất hiện trong train** — split chính thức của VQA-RAD
  vốn chia theo câu hỏi, không theo ảnh. Val theo QA pair vì vậy khớp đúng điều
  kiện test (ảnh đã thấy, câu hỏi mới). Báo cáo phải nêu phân tích này — "có
  data leakage không?" là câu hỏi vấn đáp dễ gặp.
- Test 451 mẫu chỉ dùng cho đánh giá cuối cùng.

## 5. Training & evaluation

### 5.1 Training (`train.py`)

- CLI: `python -m midterm.train --fusion concat|hadamard|cross_attention`
  → 3 lệnh = 3 thí nghiệm của báo cáo.
- Optimizer AdamW, LR `1e-3` cho phần trainable (fusion + projection + head),
  cosine decay, batch 64, tối đa 30 epoch, early stopping theo val overall
  accuracy (patience 5). Khi bật `unfreeze_last_block`: layer4 của ResNet dùng
  LR riêng `1e-5` (param group thứ hai).
- Device tự chọn: `cuda` → `mps` → `cpu`.
- **Seed toàn cục** (random/numpy/torch, cả CUDA) fix mặc định 42 ở đầu mỗi run —
  3 thí nghiệm fusion chỉ khác nhau ở fusion, không khác ở khởi tạo ngẫu nhiên.
- `run_name` mặc định = tên fusion (vd `concat`); ghi đè được bằng `--run-name`.
- Mỗi run lưu vào `outputs/<run_name>/`: `config.json`, `history.json`
  (loss/acc theo epoch), biểu đồ `curves.png` (matplotlib backend Agg);
  checkpoint tốt nhất vào `checkpoints/<run_name>.pt`.

### 5.2 Evaluation (`evaluate.py`)

- CLI: `python -m midterm.evaluate --checkpoint checkpoints/<run_name>.pt`.
- Ba số liệu chuẩn của VQA-RAD: **overall / closed (yes-no) / open accuracy**.
  Phân loại closed vs open bằng đáp án chuẩn hóa thuộc {yes, no}.
- Xuất bảng ví dụ dự đoán đúng/sai (ảnh id, câu hỏi, đáp án thật, top-1 dự đoán)
  → nguyên liệu cho phần phân tích lỗi của báo cáo.

### 5.3 Demo (`demo.py`)

- CLI: `python -m midterm.demo --checkpoint <ckpt> --image x.jpg --question "..."`.
- In top-5 đáp án kèm xác suất softmax.
- Với checkpoint `cross_attention`: lưu thêm attention map overlay lên ảnh
  (PNG) — minh họa cho báo cáo và vấn đáp.

## 6. Cấu trúc thư mục

```
midterm/
├── README.md            # cập nhật: kiến trúc, cách chạy, bảng kết quả
├── requirements.txt     # cho Colab: torch/torchvision/transformers/datasets
├── config.py            # dataclass Config: mọi hyperparameter + fusion type
├── data/
│   ├── download.py
│   ├── vocab.py
│   └── dataset.py
├── models/
│   ├── image_encoder.py
│   ├── text_encoder.py
│   ├── fusion.py
│   └── vqa_model.py
├── train.py
├── evaluate.py
├── demo.py
└── colab_train.ipynb    # notebook mỏng cho Colab: clone → install → train ×3 → evaluate
```

- Mỗi module một trách nhiệm: đổi fusion không đụng encoder, đổi encoder không
  đụng training loop. Tất cả lựa chọn đi qua `config.py`.
- Colab: `colab_train.ipynb` là notebook **all-in-one tự chứa toàn bộ logic**
  (config + data + model + train + evaluate + demo inline) — không cần `git clone`,
  không phụ thuộc các module `.py`. Quyết định này (2026-06-17, theo yêu cầu người
  dùng) thay cho thiết kế "notebook mỏng gọi CLI" ban đầu: notebook mỏng dùng
  `python -m midterm.X` phụ thuộc CWD là repo root, mà Colab reconnect làm mất
  `%cd` → `ModuleNotFoundError`. Bản all-in-one chạy mọi thứ trong process nên chỉ
  cần Run-all lại là xong. Đánh đổi: logic bị nhân đôi giữa notebook và `.py`
  (các `.py` vẫn giữ cho CLI local); khi sửa model phải đồng bộ cả hai nơi.
- Artifacts (`data/vqa_rad/`, `answer_vocab.json`, `checkpoints/`, `outputs/`)
  đã nằm trong `.gitignore`.

## 7. Rủi ro & cách xử lý

| Rủi ro | Xử lý |
|---|---|
| Overfit — chỉ 1.793 mẫu train | Freeze encoder, dropout 0.5, early stopping, augmentation nhẹ |
| Mất cân bằng yes/no (closed chiếm ~nửa) | Báo cáo closed/open riêng; phân tích lỗi theo loại câu hỏi |
| Đáp án test ngoài vocab | In độ phủ vocab, chấp nhận sai minh bạch |
| Python 3.14 local vs Colab (3.11/3.12) | Code chỉ dùng API ổn định của torch/transformers; smoke test cả hai nơi |
| Tải pretrained chậm trên Colab | Dùng cache HF mặc định; tổng dung lượng model ~500MB |

## 8. Ngoài phạm vi (YAGNI)

- Generation/seq2seq, LLM-based VQA (LLaVA-Med, BLIP).
- Ablation image/text encoder (chỉ là thí nghiệm phụ *tùy thời gian*, không cam kết).
- Hyperparameter search tự động, multi-GPU, mixed precision.
- Web UI cho demo — chỉ CLI.
