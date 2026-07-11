# Day 7 — Multimodal Fusion: kết hợp nhiều model thành một

Code: [`01_multimodal_fusion.py`](01_multimodal_fusion.py) — train một model duy
nhất nhận đồng thời **ảnh MNIST** (chữ số $d$) và **caption tiếng Việt** "cộng
$k$", dự đoán nhãn $(d + k) \bmod 10$. Sau 5 epoch, model đạt **97.97%**
accuracy trên tập test; ablation (tắt từng nhánh) chứng minh model thực sự
buộc phải dùng **cả hai** nguồn dữ liệu, không "ăn gian" bằng một nguồn duy nhất.

## 1. Bài toán và động cơ

Rất nhiều bài toán thực tế có **nhiều nguồn dữ liệu khác bản chất nhau** cùng
mô tả một đối tượng: Visual Question Answering (ảnh + câu hỏi văn bản), image
captioning (ảnh → mô tả), video hiểu ngữ cảnh (hình ảnh + âm thanh), hệ thống
giám sát (sensor số + log văn bản)... Điểm chung: mỗi nguồn (modality) mang một
phần thông tin, và câu trả lời đúng chỉ có được khi **kết hợp** chúng lại.

Cái khó là mỗi modality có cấu trúc dữ liệu khác nhau, nên **không thể dùng
chung một encoder**:

- Ảnh có cấu trúc không gian 2D (pixel gần nhau tương quan mạnh) → cần CNN,
  tận dụng tính bất biến dịch chuyển (translation invariance) của conv/pooling.
- Văn bản là một **chuỗi** ký hiệu rời rạc, độ dài thay đổi, có thứ tự → cần
  RNN/LSTM/GRU hoặc Transformer, như đã thấy ở day 6.

Không thể "làm phẳng" ảnh thành vector rồi nối với ID của từ và ném vào một
MLP — mỗi nguồn cần được **encode riêng** thành một vector đặc trưng (feature
vector) có ý nghĩa, rồi mới **hợp nhất (fusion)** các vector đó ở đâu đó trong
mạng.

Bài toán đồ chơi của day 7 được thiết kế cố ý để **buộc** model phải fusion:
ảnh MNIST chứa chữ số $d$ (0–9), caption tiếng Việt "cộng $k$" chứa số $k$
(0–9, viết bằng chữ, ví dụ "cộng ba"), nhãn là $(d + k) \bmod 10$. Quan sát
mấu chốt:

- Chỉ nhìn ảnh (không biết $k$): $d$ cố định nhưng $k$ ngẫu nhiên đều trên
  10 giá trị → nhãn $(d+k) \bmod 10$ cũng đều trên 10 giá trị → **đoán mò,
  ~10% accuracy**, dù model "nhìn ảnh rất kỹ".
- Chỉ đọc caption (không biết $d$): tương tự, chỉ còn ~10%.
- Chỉ khi có **cả hai**, $(d+k) \bmod 10$ mới xác định được chính xác.

Nói cách khác, thiết kế nhãn kiểu cộng-modulo loại bỏ hoàn toàn khả năng
"gian lận" bằng một nguồn duy nhất — đây chính là điều kiện cần để một bài
tập multimodal có ý nghĩa: nếu chỉ cần một nhánh là đủ, model sẽ học cách lờ
đi nhánh còn lại (một hiện tượng thật gọi là *modality collapse*), và ta sẽ
không kiểm chứng được gì về khả năng fusion.

## 2. Kiến trúc 1: fusion bằng concat (sơ đồ trên bảng)

```
 ảnh MNIST [1,28,28]                caption "cộng k" [2 token]
        │                                   │
        ▼                                   ▼
 ┌─────────────────┐                ┌──────────────────┐
 │   ImageEncoder   │                │    TextEncoder    │
 │  conv3x3 → ReLU  │                │  Embedding(12,16) │
 │   → maxpool 2x2  │                │        ↓          │
 │  conv3x3 → ReLU  │                │   LSTM(16 → 32)   │
 │   → maxpool 2x2  │                │  lấy hidden CUỐI  │
 │ flatten → Linear │                │                   │
 │      → ReLU      │                │                   │
 └─────────────────┘                └──────────────────┘
        │ f_img  (64 chiều)                 │ f_txt  (32 chiều)
        └───────────────────┬───────────────┘
                             ▼
                 concat  [f_img ; f_txt]  ∈ R^96      ← FUSION
                             │
                             ▼
                   ┌──────────────────┐
                   │   FC head        │
                   │ Linear(96→64)    │
                   │ ReLU             │
                   │ Linear(64→10)    │
                   └──────────────────┘
                             │
                             ▼
                    logits (10 lớp) → CrossEntropyLoss(., y)
                             │
              ◄══════════════╝  gradient lan ngược qua TOÀN BỘ sơ đồ
     optimizer.step()  ⇒  cập nhật θ_CNN, θ_LSTM, θ_head CÙNG MỘT LÚC
                          ("update parameter ở đây")
```

**Nhánh ảnh (`ImageEncoder`)**: hai tầng `Conv2d(kernel_size=3, padding=1)`.
Conv 3×3 trượt một cửa sổ 3×3 pixel qua toàn ảnh, mỗi vị trí tính một tích
chập (weighted sum) rồi cho ra một giá trị — cùng bộ trọng số dùng lại ở mọi
vị trí (weight sharing), nên conv học được các đặc trưng cục bộ (nét, cạnh,
vòng cong) bất kể chúng xuất hiện ở đâu trong ảnh. Sau mỗi conv là
`MaxPool2d(2)`: chia ảnh thành các ô 2×2, giữ lại giá trị lớn nhất mỗi ô →
kích thước không gian giảm một nửa mỗi chiều (28→14→7), đồng thời tăng dần
"tầm nhìn" (receptive field) của các tầng sau. Sau hai tầng conv+pool, bản đồ
đặc trưng `[32, 7, 7]` được `Flatten` thành vector 1568 chiều rồi qua
`Linear(1568, 64)` nén về đúng $D_{img} = 64$ chiều — đây là $f_{img}$.

**Nhánh text (`TextEncoder`)**: `Embedding` là một bảng tra cứu — mỗi token
(ví dụ từ "cộng", "ba") có một chỉ số nguyên, và bảng embedding ánh xạ chỉ số
đó sang một vector 16 chiều học được (không phải one-hot cố định). Caption
"cộng k" luôn có đúng 2 token; `LSTM` đọc lần lượt từng token, cập nhật hidden
state, và **hidden state sau token cuối cùng** được lấy làm $f_{txt}$ (32
chiều) — chính là "bản tóm tắt" của cả câu, hệt cách encoder trong bài seq2seq
ở day 6 dùng hidden cuối làm context vector $c$. Khác biệt duy nhất: ở đây câu
luôn dài 2 token nên không cần `pack_padded_sequence` hay attention qua nhiều
bước — bài toán chủ đích giữ nhánh text đơn giản để tập trung vào cơ chế fusion.

**Fusion**: $f = [f_{img} \, ; \, f_{txt}] \in \mathbb{R}^{64+32} = \mathbb{R}^{96}$
— nối hai vector đặc trưng thành một vector duy nhất bằng phép `torch.cat`.
Đây là kiểu fusion đơn giản nhất, gọi là **early fusion** hay **feature-level
fusion**: hợp nhất ngay ở mức vector đặc trưng, trước khi ra quyết định cuối
cùng (so với **late fusion** — mỗi nhánh tự dự đoán rồi mới hợp nhất kết quả,
xem mục 5).

## 3. Toán: luồng gradient qua hai nhánh

Hàm mất mát của toàn bộ model:

$$L = \text{CrossEntropy}\big(\text{head}([f_{img} \, ; \, f_{txt}]),\; y\big)$$

Điểm mấu chốt để hiểu vì sao một model duy nhất có thể học tốt cả hai nhánh
cùng lúc: `concat` là một phép **tuyến tính đơn giản theo từng thành phần** —
nó chỉ "xếp cạnh nhau", không trộn giá trị. Vì vậy khi backprop, gradient của
loss theo vector fusion $\partial L / \partial f \in \mathbb{R}^{96}$ **tách
khối một cách tự nhiên**: 64 thành phần đầu chính là $\partial L/\partial
f_{img}$, chảy thẳng về nhánh CNN; 32 thành phần sau là $\partial L/\partial
f_{txt}$, chảy thẳng về nhánh LSTM — không cần bất kỳ phép toán đặc biệt nào
để "tách" gradient, phép concat tự làm việc đó khi đảo ngược:

$$\frac{\partial L}{\partial \theta_{CNN}} = \frac{\partial L}{\partial f_{img}} \cdot \frac{\partial f_{img}}{\partial \theta_{CNN}}
\qquad\qquad
\frac{\partial L}{\partial \theta_{LSTM}} = \frac{\partial L}{\partial f_{txt}} \cdot \frac{\partial f_{txt}}{\partial \theta_{LSTM}}$$

Trong code, dòng `loss.backward()` tính toàn bộ chuỗi đạo hàm này chỉ bằng
một lệnh — autograd tự dò lại đồ thị tính toán head → concat → CNN/LSTM. Dòng
tiếp theo, `optimizer.step()`, mới là **"update parameter ở đây"** trên bảng:
`torch.optim.Adam(model.parameters(), ...)` được khởi tạo với tham số của
*toàn bộ* model (`img_enc` + `txt_enc` + `head`), nên một lệnh `step()` duy
nhất cập nhật đồng thời $\theta_{CNN}$, $\theta_{LSTM}$, $\theta_{head}$.

Đáng để so sánh với transfer learning ở **day 5**: ở đó, features được trích
**một lần**, trong `torch.no_grad()` (backbone hoàn toàn ngoài vòng backward),
rồi cache lại dưới dạng tensor thô — training loop chỉ học trên những features
đã đông cứng đó, không có gradient nào qua backbone cả. Ở day 7, ngược lại:
gradient chảy **end-to-end** qua cả CNN lẫn LSTM, một lệnh `optimizer.step()`
duy nhất cập nhật mọi tham số cùng lúc, đồng lúc chịu áp lực từ một loss duy
nhất. Đây là **train end-to-end** đúng nghĩa, khác hẳn cách "train từng nhánh
riêng rồi ghép" (ví dụ: train sẵn một classifier ảnh, một classifier text,
rồi mới ghép feature — cách này không có gradient chung nên hai nhánh không
"biết" thương lượng với nhau).

## 4. Ablation: bằng chứng model dùng cả hai nguồn

Train xong không có nghĩa là model *thực sự* dùng cả hai nhánh — biết đâu
nhánh ảnh mạnh đến mức nhánh text bị bỏ xó (dù thiết kế nhãn ở mục 1 đã cố
tình ngăn điều này). Cách kiểm chứng trực tiếp nhất: **ablation** — sau khi
train xong, ở bước eval, thay vector đặc trưng của một nhánh bằng **vector 0**
(tham số `drop_img=True` / `drop_txt=True` trong `MultimodalNet.forward`),
rồi đo lại accuracy trên tập test. Nếu accuracy tụt về mức đoán mò, nghĩa là
nhánh bị tắt thực sự mang thông tin không thể thiếu.

Bảng kết quả THẬT từ lần chạy (5 epoch, `python day7/01_multimodal_fusion.py`):

| Chế độ | Test accuracy |
|---|---|
| Đủ cả hai nhánh | **0.9797** |
| Tắt nhánh ảnh (`drop_img`, chỉ còn text) | **0.1061** |
| Tắt nhánh text (`drop_txt`, chỉ còn ảnh) | **0.0985** |

Cả hai trường hợp ablation đều rơi xuống quanh **~10%** — đúng như dự đoán ở
mục 1 bằng lý thuyết thông tin: khi thiếu $d$ (tắt ảnh) hoặc thiếu $k$ (tắt
text), nhãn $(d+k) \bmod 10$ trở thành một biến ngẫu nhiên **đều (uniform)**
trên 10 giá trị đối với phần thông tin còn lại — không có mẫu hình nào để
model khai thác, nên chiến lược tốt nhất cũng chỉ là đoán ngẫu nhiên giữa 10
lớp, tức chính xác 1/10 = 10%. Việc accuracy **đủ hai nhánh (97.97%)** cao
hơn hẳn hai mức tắt nhánh (10.61% và 9.85%) — gần bằng khoảng cách lý
thuyết tối đa có thể — là bằng chứng thực nghiệm rằng model đã học cách **kết
hợp thông tin từ cả hai nguồn** để giải bài toán, chứ không chỉ dựa vào một
nhánh và "trúng may" nhờ nhánh kia.

## 5. Các chiến lược fusion khác (hướng mở rộng)

Concat là fusion đơn giản nhất nhưng không phải duy nhất. Vài hướng phổ biến:

- **Late fusion**: mỗi nhánh có head riêng, tự ra một dự đoán (logits hoặc
  xác suất) độc lập, rồi hợp nhất *ở tầng quyết định* — ví dụ trung bình cộng
  hai vector logits, hoặc vote. Ưu điểm: hai nhánh có thể train/đánh giá tách
  rời; nhược điểm: mất khả năng học các tương tác *phi tuyến giữa hai
  modality* trước khi ra quyết định (xem bài tập 2 — vì sao bài toán ngày 7
  vẫn giải được bằng late fusion).
- **Fusion bằng phép nhân/gating**: thay vì nối cạnh nhau, dùng một phép nhân
  từng phần tử có điều kiện, ví dụ $f_{img} \odot (W \cdot f_{txt})$ — nhánh
  text đóng vai trò "cổng" (gate) điều chỉnh nhánh ảnh, cho phép mô hình hoá
  tương tác phi tuyến mạnh hơn cộng/nối đơn thuần (ý tưởng nền tảng của các
  cơ chế gating như trong LSTM/GRU chính nó).
- **Attention/cross-attention fusion**: thay vì tóm gọn mỗi nhánh về đúng một
  vector rồi ghép, để text "hỏi" ảnh — dùng embedding của caption làm query,
  các vùng đặc trưng không gian của ảnh (feature map, chưa flatten) làm
  key/value, tính attention để mô hình tự chọn "nhìn vào đâu trên ảnh ứng với
  từng từ". Đây là hướng dẫn thẳng tới các kiến trúc multimodal Transformer
  hiện đại — VQA attention-based, CLIP (đối chiếu embedding ảnh/text trong
  cùng không gian), các mô hình vision-language lớn — vượt xa phạm vi bài
  toán đồ chơi ở đây nhưng đáng để biết tên và ý tưởng cốt lõi.

## 6. Bài tập về nhà

1. **Cộng thay vì concat**: chiếu $f_{img}$ và $f_{txt}$ về cùng số chiều
   (ví dụ cả hai về 32 chiều bằng một `Linear`), rồi cộng thay vì nối
   (`f_img + f_txt`). Accuracy thay đổi thế nào so với concat? Gợi ý: cộng mất
   thông tin "thành phần nào tới từ nhánh nào" mà concat giữ nguyên — head sau
   đó phải tự suy luận nhiều hơn từ một không gian chung.
2. **Late fusion**: tạo hai head riêng — một head chỉ nhận $f_{img}$, một head
   chỉ nhận $f_{txt}$ — rồi **cộng hai logits** lại trước softmax. Vì sao bài
   này late fusion vẫn giải được, dù mỗi nhánh riêng lẻ "không đủ thông tin"?
   Gợi ý: $(d+k) \bmod 10$ có cấu trúc **cộng** — nếu mỗi head học cách xuất
   ra một phân phối phù hợp (kiểu "dịch chuyển vòng" theo giá trị nhánh đó
   biết), tổng hai logit có thể tái tạo lại phép cộng modulo mà không cần
   tương tác phi tuyến giữa hai nhánh.
3. **Nhiễu trong caption**: với xác suất 10%, đổi $k$ trong caption thành một
   số ngẫu nhiên khác (caption "nói dối"). Accuracy trần (đạt được sau khi
   train hội tụ) còn bao nhiêu? Có thể suy ra cận trên lý thuyết từ tỉ lệ
   nhiễu không?
4. **Bỏ LSTM**: thay `TextEncoder` bằng mean-pooling embedding đơn giản (lấy
   trung bình cộng vector embedding của 2 token thay vì cho qua LSTM). Với
   câu chỉ 2 token cố định "cộng $k$", LSTM có thực sự cần thiết không? Thử
   nghiệm và so sánh accuracy, số tham số, thời gian train.

## Chạy code

```bash
source .venv/bin/activate
python day7/01_multimodal_fusion.py
```

Kết quả một lần chạy thật (CPU):

```
thiết bị: cpu | tham số: 118,666
epoch  1/5 | train loss 1.4364 | test acc 0.9331
epoch  2/5 | train loss 0.2045 | test acc 0.9667
epoch  3/5 | train loss 0.1220 | test acc 0.9747
epoch  4/5 | train loss 0.0897 | test acc 0.9788
epoch  5/5 | train loss 0.0722 | test acc 0.9797

--- Ablation trên tập test ---
đủ cả hai nhánh    : 0.9797
tắt nhánh ảnh      : 0.1061  (kỳ vọng ~0.10 — chỉ còn text)
tắt nhánh text     : 0.0985  (kỳ vọng ~0.10 — chỉ còn ảnh)
```

Loss giảm đều và nhanh (1.4364 → 0.0722 chỉ trong 5 epoch) vì bài toán tuy đòi
hỏi fusion nhưng bản thân mỗi nhánh (nhận diện chữ số MNIST, đọc 2 token cố
định) đều là những bài con dễ; test accuracy tăng đơn điệu và đã vượt 93% ngay
từ epoch đầu tiên.

![loss](multimodal_loss_acc.png)

Biểu đồ trái: train loss giảm dần theo epoch. Biểu đồ phải: test accuracy
tăng dần, với đường ngang nét đứt tại 10% đánh dấu mức "đoán mò" — chính là
mức mà hai cấu hình ablation (tắt ảnh, tắt text) rơi xuống, trong khi model
đầy đủ vượt xa lên gần 98%.
