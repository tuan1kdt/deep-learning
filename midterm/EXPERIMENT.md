# Thực nghiệm MedVQA trên VQA-RAD — giải thích chi tiết & lý do thiết kế

Tài liệu này giải thích **toàn bộ thực nghiệm** của đồ án giữa kỳ: từ bài toán,
dữ liệu, kiến trúc model, đến quy trình huấn luyện và đánh giá. Với **mỗi lựa
chọn** đều kèm phần *“vì sao chọn như vậy”* — dùng được làm tài liệu vấn đáp.

> Tổng quan ngắn gọn + cách chạy: xem [`README.md`](README.md).
> Thiết kế gốc (đã duyệt brainstorming): [`docs/superpowers/specs/2026-06-11-medvqa-design.md`](../docs/superpowers/specs/2026-06-11-medvqa-design.md).

---

## 1. Bài toán

**Medical Visual Question Answering (MedVQA):** cho một ảnh y khoa (X-quang / CT /
MRI) và một câu hỏi tiếng Anh, model phải trả lời. Ví dụ:

> Ảnh CT bụng + *“Is the liver enlarged?”* → `no`
> Ảnh X-quang ngực + *“What organ system is imaged?”* → `chest`

### 1.1 Mô hình hóa thành classification, không phải generation

Đáp án được dự đoán bằng cách **chọn 1 class trong answer vocab cố định (429
class)**, chứ không sinh chuỗi tự do (seq2seq/decoder).

| | Classification (đã chọn) | Generation |
|---|---|---|
| Output | argmax trên 429 class | sinh token tự hồi quy |
| Eval | so khớp chính xác, dễ đo accuracy | cần BLEU/đối chiếu mềm, mơ hồ |
| Dữ liệu cần | hợp với 1.793 mẫu | đói dữ liệu, dễ học vẹt |
| Phù hợp đồ án | tự xây fusion + head là đủ | cần decoder lớn, ngoài phạm vi |

**Lý do:** đáp án VQA-RAD phần lớn ngắn và lặp lại (`yes/no`, tên cơ quan, vị
trí…), nên không gian đáp án hữu hạn — classification vừa khớp bản chất dữ liệu,
vừa cho phép đo accuracy rõ ràng và train ổn định trên tập nhỏ. Generation (LLaVA-Med,
BLIP) bị **đưa ra ngoài phạm vi** vì đói dữ liệu và lệch trọng tâm đồ án (trọng tâm
là **so sánh chiến lược fusion**, không phải sinh văn bản).

### 1.2 Tiêu chí thành công (theo thứ tự ưu tiên)

1. **Code sạch, hiểu sâu** — đọc top-to-bottom như tutorial, giải thích được mọi
   thành phần. Accuracy cao *không* phải ưu tiên số một.
2. **Báo cáo + thí nghiệm so sánh** — trục ablation chính: **fusion strategy**
   (3 biến thể trên cùng codebase, chỉ đổi 1 dòng config).
3. **Demo chạy được** — ảnh + câu hỏi vào, đáp án ra.

Kỳ vọng overall accuracy ~**50–65%** trên test với encoder pretrained bị freeze.

---

## 2. Dữ liệu — VQA-RAD

[VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad): **1.793 QA
pairs train / 451 test**, mỗi mẫu có `image`, `question`, `answer`.

### 2.1 Những con số phải biết trước khi đọc kết quả

| Con số | Giá trị | Hệ quả thiết kế |
|---|---|---|
| Ảnh duy nhất trong train | **313** | ~5,7 câu hỏi/ảnh — dữ liệu rất nhỏ, dễ overfit |
| Ảnh test trùng với train | **202/203** | Split chính thức chia theo **câu hỏi**, không theo ảnh |
| Answer vocab (từ train) | **429 class** | Mỗi đáp án duy nhất sau chuẩn hóa = 1 class |
| Độ phủ vocab trên test | **334/451 = 74,1%** | **Trần accuracy khả dĩ** — không thể vượt qua bằng model |
| Câu hỏi yes/no (closed) | **~52%** | Phải báo cáo closed/open riêng |

Đây là những con số quyết định gần như mọi lựa chọn phía sau: tập **nhỏ** → freeze
+ regularize mạnh; **trần 74,1%** → đặt kỳ vọng accuracy thực tế; **closed ~52%** →
overall accuracy dễ gây ảo tưởng nếu không tách closed/open.

### 2.2 Answer vocab — `data/vocab.py`

```python
answers = sorted({normalize_answer(a) for a in ds["train"]["answer"]})
vocab = {answer: idx for idx, answer in enumerate(answers)}   # 429 class
```

- **Build từ train split duy nhất.** Vocab không bao giờ “nhìn” test → không rò rỉ.
- **`sorted()`** để vocab ổn định giữa các lần build (set của Python không cố định
  thứ tự) → index → class lặp lại được, checkpoint tái dùng được.
- **Không có token `<unk>`.** *Vì sao?* Mọi đáp án train đều nằm trong vocab theo
  cách build, nên `<unk>` sẽ là **class chết** không bao giờ là target → chỉ làm
  loãng softmax. Đáp án test ngoài vocab được xử lý bằng cách khác (mục 2.5).
- **Chuẩn hóa đáp án** (`normalize_answer`): lowercase → bỏ khoảng trắng/dấu câu
  thừa hai đầu → gộp khoảng trắng liên tiếp. `'Yes.'`, `' yes '`, `'YES'` đều thành
  `'yes'`. *Vì sao?* Không chuẩn hóa thì `'Yes'` và `'yes.'` thành 2 class khác
  nhau → phình vocab giả tạo và phạt model vì khác biệt vô nghĩa.

### 2.3 Ảnh — `data/dataset.py`

```python
transforms.Compose([resize, ToTensor(), Normalize(IMAGENET_MEAN, IMAGENET_STD)])
```

- **Resize 224×224.** Kích thước chuẩn ResNet/ImageNet.
- **Normalize bằng mean/std ImageNet** `[0.485,0.456,0.406] / [0.229,0.224,0.225]`.
  *Vì sao bắt buộc đúng bộ số này?* ResNet được pretrain với chính phân phối đã
  chuẩn hóa đó; dùng số khác sẽ đẩy ảnh ra khỏi miền backbone từng thấy → feature
  pretrained xuống cấp.
- **Augmentation chỉ cho train**, tắt được qua `cfg.augment`: `RandomResizedCrop`
  nhẹ (scale 0.9–1.0, ratio 1.0) — chỉ zoom/crop nhẹ, không bóp méo.
- **KHÔNG horizontal flip.** *Vì sao?* Ảnh y khoa có **tính trái/phải về giải
  phẫu** (tim bên trái, gan bên phải). Flip ngang tạo ra ảnh sai giải phẫu — một
  câu hỏi “bên nào” sẽ có nhãn ngược → augmentation phá nhãn thay vì giúp model.

### 2.4 Câu hỏi

- Tokenize bằng **BERT tokenizer** (`bert-base-uncased`), `padding="max_length"`,
  `truncation=True`, **`max_len = 32`**. *Vì sao 32?* Câu hỏi VQA-RAD ngắn (vài
  từ tới một câu); 32 token thừa sức chứa gần hết, mà vẫn giữ batch gọn.

### 2.5 Đáp án ngoài vocab → label `-1`

```python
label = self.vocab.get(answer, -1)   # ngoài vocab (chỉ ở test) → -1
```

`argmax` trên 429 class không bao giờ ra `-1`, nên evaluate **tự động tính sai**
các đáp án test ngoài vocab. *Vì sao làm vậy thay vì bỏ chúng?* Để báo cáo **minh
bạch**: 26% test không thể đúng về nguyên tắc; giấu chúng đi sẽ thổi phồng
accuracy. Trần 74,1% được in ra rõ ràng.

### 2.6 Validation split — lựa chọn có chủ đích, không phải sơ suất

```python
split = ds["train"].train_test_split(test_size=0.10, seed=42)  # THEO QA PAIR
```

VQA-RAD **không có val chính thức** → ta tách 10% từ train. Tách **theo QA pair**
(không theo ảnh), seed 42.

> **“Đây có phải data leakage không?”** — câu hỏi vấn đáp kinh điển. **Trả lời:
> Không, và đây là lựa chọn cố ý.** 202/203 ảnh test cũng xuất hiện trong train:
> split chính thức của VQA-RAD vốn chia **theo câu hỏi**, không theo ảnh. Vì vậy
> điều kiện test thật là *“ảnh đã thấy, câu hỏi mới”*. Tách val theo QA pair khớp
> đúng điều kiện đó → val là ước lượng trung thực cho test. Nếu tách **theo ảnh**,
> val sẽ **khó hơn** test thật → early stopping & chọn model bị lệch.

**Test 451 mẫu chỉ đụng tới ở bước đánh giá cuối cùng** — không dùng để tune gì.

---

## 3. Kiến trúc model

```
Ảnh y khoa ─→ ResNet-50 (pretrained, freeze) ─┬─→ v_img   (B, 768) ─────────┐
                                              └─→ img_map (B, 49, 768) ──────┤
                                                                            ├─→ [FUSION] ─→ MLP head ─→ logits (429)
Câu hỏi    ─→ BERT-base (pretrained, freeze) ───→ v_txt   (B, 768) ──────────┘
```

Nguyên tắc xuyên suốt: **encoder pretrained bị freeze, chỉ tự xây fusion + head.**
*Vì sao?* (1) Đề cho phép dùng pretrained encoder, trọng tâm là tự xây phần kết
hợp; (2) tập 1.793 mẫu quá nhỏ để fine-tune ResNet+BERT (~134M tham số) mà không
overfit nặng; (3) freeze → chỉ ~5,6–6,7M tham số trainable → train trên T4 vài
phút/epoch.

### 3.1 Image encoder — `models/image_encoder.py`

```python
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
self.backbone = nn.Sequential(*list(resnet.children())[:-2])   # bỏ avgpool + fc
```

- **ResNet-50 ImageNet (`IMAGENET1K_V2`).** *Vì sao ResNet-50?* CNN trưởng thành,
  feature ImageNet chuyển giao tốt sang ảnh y khoa, nhẹ vừa đủ cho T4. `V2` là
  bộ weight huấn luyện tốt hơn `V1`.
- **Bỏ 2 layer cuối** (`avgpool` + `fc`), giữ phần conv → feature map `(B, 2048, 7, 7)`.
- Xuất **2 dạng đặc trưng** từ cùng feature map:
  - `v_img` = average-pool toàn cục → `(B, 2048)` → `Linear 2048→768`. Dùng cho
    `concat` / `hadamard`.
  - `img_map` = flatten 7×7 → **49 vector vùng** `(B, 49, 2048)` → `Linear 2048→768`.
    Dùng cho `cross_attention`.
  *Vì sao xuất cả hai?* Để 3 fusion dùng chung một encoder; cross-attention cần
  thông tin **không gian** (vùng nào của ảnh liên quan câu hỏi) mà vector toàn cục
  đã làm mất.
- **Hai `Linear` projection riêng** đưa cả hai về `d_model = 768`. *Vì sao 768?*
  Bằng chiều ẩn BERT → fusion ghép hai modality cùng chiều, không cần co giãn lệch.
- **BatchNorm LUÔN ở eval mode** — override `train()`:
  ```python
  def train(self, mode=True):
      super().train(mode)
      self.backbone.eval()   # BN running stats đóng băng kể cả lúc train
  ```
  > **Điểm tinh tế quan trọng cho vấn đáp.** `requires_grad=False` chỉ chặn
  > optimizer cập nhật **weight**. Nhưng nếu gọi `model.train()` lan vào backbone,
  > BatchNorm vẫn cập nhật **running mean/var** theo từng batch ảnh y khoa →
  > encoder “frozen” **âm thầm đổi hành vi**, kết quả không tái lập. Ghim BN ở
  > `eval()` chặn điều đó. Gradient vẫn chảy bình thường qua module eval (cần khi
  > bật `unfreeze_last_block`).
- **`unfreeze_last_block`** (mặc định `False`): mở `layer4` (block conv sâu nhất)
  làm thí nghiệm phụ, train với LR riêng `1e-5` (mục 4).

### 3.2 Text encoder — `models/text_encoder.py`

```python
self.bert = AutoModel.from_pretrained("bert-base-uncased")   # freeze
```

- **`bert-base-uncased`.** *Vì sao?* Câu hỏi VQA-RAD là tiếng Anh thường; BERT-base
  đủ mạnh, nhẹ, phổ biến. (Tên model là config → đổi sang `Bio_ClinicalBERT` là
  thí nghiệm phụ tùy thời gian.)
- **Mean-pooling hidden state lớp cuối (có mask)** thay vì `[CLS]`:
  ```python
  mask = attention_mask.unsqueeze(-1)
  v_txt = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # bỏ padding
  ```
  > **Vì sao mean-pooling chứ không `[CLS]`?** Vector `[CLS]` của BERT được
  > pretrain cho **next-sentence prediction**; khi **không fine-tune**, nó là biểu
  > diễn câu **yếu**. Mean-pooling các token thật (loại padding) thường tốt hơn rõ
  > rệt với BERT freeze. `--text-pool cls` được giữ làm **đối chứng** rẻ tiền.
- **Freeze toàn bộ + luôn `eval()`** (override `train()` như image encoder).
  *Vì sao eval?* Tắt dropout bên trong BERT → cùng câu hỏi luôn cho cùng vector
  giữa các batch → biểu diễn ổn định, kết quả tái lập.

### 3.3 Fusion — trục ablation chính (`models/fusion.py`)

Cả 3 cùng interface `forward(v_img, img_map, v_txt) → (fused 768-d, attn|None)`,
nên **đổi fusion = đổi 1 dòng config**, không đụng phần còn lại.

| Tên | Công thức | Ý tưởng | Dùng spatial? |
|---|---|---|---|
| `concat` | `ReLU(Linear([v_img ; v_txt]))` | Baseline đơn giản nhất | Không |
| `hadamard` | `proj_img(v_img) ⊙ proj_txt(v_txt)` | Tương tác **nhân** từng chiều | Không |
| `cross_attention` | `LayerNorm(v_txt + MHA(Q=v_txt, K=V=img_map))` | Câu hỏi “nhìn” 49 vùng ảnh | **Có** |

- **`concat`** — *vì sao là baseline:* nối hai vector rồi để một `Linear` tự học
  cách trộn. Không giả định gì về cách hai modality tương tác → mốc tham chiếu.
- **`hadamard`** — *vì sao thử nhân từng phần tử:* phép `⊙` buộc hai modality
  “đồng thuận” theo từng chiều — feature chỉ lớn khi **cả ảnh lẫn câu hỏi** cùng
  kích hoạt chiều đó. Đây là cơ chế kinh điển trong VQA (MCB/MLB) để mô hình hóa
  tương tác bậc hai mà concat (chỉ cộng tuyến tính) không nắm được.
- **`cross_attention`** — *vì sao mạnh nhất về lý thuyết:* câu hỏi là **query duy
  nhất** nhìn vào **49 vùng ảnh** (key/value) bằng multi-head attention (8 head),
  cộng **residual** với `v_txt` rồi **LayerNorm** — một block Transformer tối giản.
  Model học **chú ý vào vùng ảnh liên quan câu hỏi** thay vì nén cả ảnh thành 1
  vector. Bonus: attention weights `(B, 49)` **trực quan hóa được** → heatmap demo.

> **⚠️ Confound phải thừa nhận trong báo cáo.** `cross_attention` được cấp **49
> vùng spatial** mà `concat`/`hadamard` **không** dùng. Vì thế chênh lệch kết quả
> **trộn lẫn hai yếu tố**: (a) cơ chế fusion (attention vs concat/nhân) và (b) có
> thêm thông tin không gian. Không thể quy hết khác biệt cho “attention tốt hơn”.
> Gỡ confound (vd cho concat dùng cả 49 vùng pooled) nằm ngoài phạm vi đồ án,
> nhưng **kết luận phải nêu rõ giới hạn này**.

### 3.4 Classifier head — `models/vqa_model.py`

```python
nn.Sequential(
    nn.Linear(768, 1024), nn.ReLU(),
    nn.Dropout(0.5),                    # chống overfit trên ~1.6k mẫu
    nn.Linear(1024, 429),
)
```

- **MLP 1 lớp ẩn (768→1024→429)** + ReLU. *Vì sao 1 lớp ẩn:* đủ phi tuyến để
  ánh xạ vector fused → 429 class, không sâu quá mức cho tập nhỏ.
- **Dropout 0.5** ngay trước lớp phân loại. *Vì sao mạnh tới 0.5:* tập train chỉ
  ~1.6k mẫu, dropout cao là tuyến phòng thủ overfit chính ở phần trainable.
- **Loss = CrossEntropyLoss** (chuẩn cho classification một nhãn).

**Tổng tham số:** ~**134M tổng** (gần hết là ResNet+BERT freeze) | chỉ
**~5,6–6,7M trainable** (fusion + 2 projection + head) tùy fusion.

---

## 4. Huấn luyện — `train.py`

### 4.1 Ba thí nghiệm của báo cáo

```bash
python -m midterm.train --fusion concat
python -m midterm.train --fusion hadamard
python -m midterm.train --fusion cross_attention
```

Chỉ khác **một** biến: `fusion`. Mọi thứ khác (data, seed, optimizer, epoch…) giữ
nguyên → so sánh **công bằng**, chênh lệch quy được về fusion (trừ confound 3.3).

### 4.2 Optimizer & learning rate — 2 param group

```python
param_groups = [{"params": new_params, "lr": 1e-3}]          # fusion + proj + head
if backbone_params:                                          # chỉ khi unfreeze
    param_groups.append({"params": backbone_params, "lr": 1e-5})
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
```

- **AdamW** — Adam + weight decay tách rời (regularize đúng cách). `weight_decay=1e-2`.
- **LR `1e-3` cho phần tự xây.** Phần này khởi tạo **ngẫu nhiên** → cần LR đủ lớn
  để học nhanh từ đầu.
- **LR `1e-5` cho `layer4` ResNet khi unfreeze** (nhỏ hơn **100 lần**). *Vì sao
  tách group:* feature pretrained đã tốt; gradient lớn lúc đầu sẽ **phá** chúng.
  LR nhỏ chỉ tinh chỉnh nhẹ. (Mặc định không unfreeze nên group này thường rỗng.)
- **Cosine annealing** giảm LR mượt về 0 theo 30 epoch — hội tụ ổn định, không cần
  dò lịch LR thủ công.

### 4.3 Vòng lặp & dừng sớm

| Hyperparameter | Giá trị | Lý do |
|---|---|---|
| `batch_size` | 64 | Vừa VRAM T4, gradient đủ mượt |
| `max_epochs` | 30 | Trần trên; thực tế thường dừng sớm hơn |
| `patience` | 5 | Early stopping theo **val overall accuracy** |
| `seed` | 42 | Fix random/numpy/torch(+CUDA) đầu mỗi run |

- **Early stopping theo val accuracy** (không phải val loss): tiêu chí đánh giá
  cuối là accuracy → chọn model trực tiếp theo nó. Lưu **checkpoint tốt nhất**, nếu
  5 epoch liền không cải thiện thì dừng → tiết kiệm và tránh overfit cuối.
- **Seed cố định:** *vì sao quan trọng ở đây:* 3 thí nghiệm fusion phải chỉ khác ở
  fusion, **không** khác ở khởi tạo trọng số hay thứ tự shuffle → seed chung đảm
  bảo điều đó.
- **`model.train()` mỗi epoch** nhưng encoder **tự ghim eval bên trong** (mục 3.1,
  3.2) → BN/dropout của backbone không bao giờ bật.

### 4.4 Tổng hợp chiến lược chống overfit

Tập chỉ ~1.6k mẫu train, nên overfit là rủi ro số một. Bốn tuyến phòng thủ:

1. **Freeze encoder** → chỉ ~6M tham số trainable (thay vì ~134M).
2. **Dropout 0.5** ở head.
3. **Weight decay 1e-2** (AdamW).
4. **Augmentation nhẹ** (random resized crop 0.9–1.0) + **early stopping** patience 5.

### 4.5 Artifact mỗi run

`outputs/<run_name>/` chứa `config.json`, `history.json` (loss/acc theo epoch),
`curves.png`; checkpoint tốt nhất ở `checkpoints/<run_name>.pt` (kèm config +
num_classes để dựng lại model). `run_name` mặc định = tên fusion.

Smoke test local (Mac MPS, ~2 phút): `--smoke` chạy subset 128 mẫu / 2 epoch chỉ
để xác nhận pipeline chạy end-to-end, **không** dùng để báo cáo.

---

## 5. Đánh giá — `evaluate.py`

Chỉ chạy ở **bước cuối** trên test 451 mẫu:

```bash
python -m midterm.evaluate --checkpoint midterm/checkpoints/concat.pt
```

### 5.1 Ba số liệu chuẩn của VQA-RAD

| Metric | Định nghĩa | Vì sao cần |
|---|---|---|
| **overall** | accuracy toàn bộ test | con số tổng |
| **closed** | accuracy trên câu hỏi yes/no (đáp án ∈ {yes, no}) | ~52% test, dễ đoán mò 50% |
| **open** | accuracy phần còn lại | khó hơn nhiều, phản ánh năng lực thật |

> **Vì sao bắt buộc tách closed/open?** Closed chiếm ~nửa test và một model đoán
> mò yes/no đã được ~50% closed. Nếu chỉ nhìn **overall**, một model giỏi yes/no
> nhưng dốt câu hỏi mở trông vẫn “ổn” → **ảo tưởng**. Tách ra mới thấy năng lực
> thật trên phần open khó.

- **Độ phủ vocab trên test (74,1%)** được in kèm như **trần accuracy** — đặt mọi
  con số vào đúng ngữ cảnh.
- **Xuất bảng ví dụ đúng/sai** + `test_results.json` (đầy đủ question / answer /
  pred / correct) → **nguyên liệu phân tích lỗi** cho báo cáo.

### 5.2 Kết quả

> ⏳ **Chưa điền** — `checkpoints/` và `outputs/` hiện trống. Điền sau khi train
> trên Colab GPU (`colab_train.ipynb`), số lấy từ `python -m midterm.evaluate`.

| Fusion | Overall | Closed (yes/no) | Open |
|---|---|---|---|
| concat | — | — | — |
| hadamard | — | — | — |
| cross_attention | — | — | — |

*(Trần accuracy khả dĩ ≈ 74,1% do giới hạn độ phủ vocab.)*

---

## 6. Demo — `demo.py`

```bash
python -m midterm.demo --checkpoint midterm/checkpoints/cross_attention.pt \
    --image x.jpg --question "is there cardiomegaly?"
```

- In **top-5 đáp án kèm xác suất softmax** — cho thấy model phân vân giữa những
  đáp án nào, không chỉ top-1.
- Với checkpoint **`cross_attention`**: vẽ **heatmap attention** (49 vùng 7×7 phóng
  to 32× bằng `np.kron`) **chồng lên ảnh** → minh họa model “nhìn” vào đâu khi trả
  lời. Đây là bằng chứng trực quan cho sức mạnh của cross-attention trong báo cáo.

`load_model()` dựng lại kiến trúc **từ config lưu trong checkpoint**, nên
demo/evaluate không cần biết run đó dùng fusion hay text_pool nào.

---

## 7. Hai cách chạy: CLI module vs notebook all-in-one

| | CLI (`python -m midterm.X`) | `colab_train.ipynb` |
|---|---|---|
| Dùng khi | Local / smoke test | Train thật trên Colab GPU |
| Cấu trúc | Module tách biệt (config/data/models/…) | **Tất-cả-trong-một**, logic inline |
| Phụ thuộc | CWD = repo root | Không (chạy mọi thứ trong process) |

> **Vì sao notebook tự chứa toàn bộ logic thay vì gọi CLI?** Notebook “mỏng” dùng
> `python -m midterm.X` phụ thuộc CWD là repo root; khi **Colab reconnect** mất
> `%cd` → `ModuleNotFoundError: No module named 'midterm'`. Bản all-in-one chạy mọi
> thứ trong process nên chỉ cần **Run all** lại là xong.
> **Đánh đổi:** logic bị **nhân đôi** giữa notebook và các `.py` — sửa model phải
> đồng bộ cả hai nơi.

---

## 8. Phạm vi: cố tình KHÔNG làm (YAGNI)

- Generation / seq2seq, LLM-based VQA (LLaVA-Med, BLIP) — đói dữ liệu, lệch trọng tâm.
- Ablation encoder ảnh/text (chỉ là thí nghiệm phụ *tùy thời gian*).
- Auto hyperparameter search, multi-GPU, mixed precision.
- Web UI cho demo — chỉ CLI.

---

## 9. Bộ câu hỏi vấn đáp nhanh

| Câu hỏi | Trả lời ngắn |
|---|---|
| Sao là classification không phải generation? | §1.1 — đáp án ngắn lặp lại, đo accuracy rõ, hợp tập nhỏ |
| Sao freeze encoder? | §3 — 1.6k mẫu quá ít để fine-tune 134M tham số; đề cho dùng pretrained |
| Sao BatchNorm ghim eval? | §3.1 — `requires_grad=False` không chặn running stats; tránh encoder âm thầm trôi |
| Sao mean-pool thay vì `[CLS]`? | §3.2 — `[CLS]` của BERT freeze là biểu diễn câu yếu |
| Val split theo QA pair có leakage không? | §2.6 — không; cố ý khớp điều kiện test (ảnh đã thấy, câu hỏi mới) |
| Sao bỏ `<unk>`? | §2.2 — sẽ là class chết, không bao giờ làm target |
| Sao không horizontal flip? | §2.3 — ảnh y khoa có tính trái/phải về giải phẫu |
| Confound trong ablation fusion là gì? | §3.3 — chỉ cross_attention dùng 49 vùng spatial |
| Sao tách closed/open? | §5.1 — closed ~52% dễ đoán mò, overall gây ảo tưởng |
| Trần accuracy ~74% từ đâu? | §2.1/§2.5 — 26% đáp án test ngoài vocab, chắc chắn sai |
