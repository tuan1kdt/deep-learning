# Day 6 — Dịch máy với Encoder-Decoder (Seq2Seq) và Attention

Code: [`01_seq2seq_attention_translation.py`](01_seq2seq_attention_translation.py) —
train model dịch Anh→Việt trên ~11k cặp câu Tatoeba, đạt BLEU-4 ≈ 20–21 trên các
câu chưa từng thấy sau ~2 phút train trên Mac (MPS).

## 1. Bài toán dịch máy — tại sao cần kiến trúc mới?

Dịch máy là bài toán **sequence-to-sequence**: đầu vào là một chuỗi (câu tiếng
Anh), đầu ra cũng là một chuỗi (câu tiếng Việt). Ba đặc điểm khiến MLP/CNN của
các bài trước không dùng thẳng được:

1. **Độ dài vào/ra thay đổi** — MLP cần vector đầu vào cố định; câu thì lúc 3 từ
   lúc 12 từ, và câu dịch ra thường *khác độ dài* câu nguồn.
2. **Không có alignment 1-1** — từ thứ i của câu đích không tương ứng cứng với từ
   thứ i của câu nguồn ("dog food" → "thức ăn cho chó": đảo trật tự, 2 từ thành 4).
3. **Phụ thuộc xa** — từ cuối câu đích có thể phụ thuộc từ đầu câu nguồn.

Ta cần model **đọc hết** câu nguồn rồi **sinh dần** câu đích, mỗi từ sinh ra được
điều kiện hoá trên cả câu nguồn lẫn các từ đích đã sinh:

$$P(y_1, \dots, y_{T'} \mid x_1, \dots, x_T) = \prod_{t=1}^{T'} P(y_t \mid y_{<t},\, x)$$

## 2. Encoder-Decoder thuần (Sutskever et al., 2014)

Ý tưởng gốc gồm 2 mạng hồi quy (RNN/GRU/LSTM):

```
  the   dog   runs        <sos>  con   chó   chạy
   │     │     │            │     │     │     │
   ▼     ▼     ▼            ▼     ▼     ▼     ▼
 ┌───────────────┐   c    ┌─────────────────────┐
 │    ENCODER    │ ─────▶ │       DECODER       │
 └───────────────┘        └─────────────────────┘
                            │     │     │     │
                            ▼     ▼     ▼     ▼
                           con   chó   chạy <eos>
```

- **Encoder** đọc từng từ nguồn, cập nhật hidden state; hidden state *cuối cùng*
  $c$ được coi là "bản tóm tắt" của cả câu (context vector).
- **Decoder** nhận $c$ làm hidden state khởi đầu, bắt đầu từ token `<sos>`,
  sinh từng từ cho đến khi tự phát ra `<eos>`.

**Điểm nghẽn:** *toàn bộ* thông tin câu nguồn phải chen chúc qua đúng **một
vector cố định** $c$ (ở đây 256 chiều). Câu càng dài, thông tin đầu câu càng bị
"pha loãng" — thực nghiệm của Bahdanau cho thấy BLEU của seq2seq thuần rơi rõ
rệt khi câu vượt ~20–30 từ.

## 3. Attention (Bahdanau et al., 2015) — cho decoder "nhìn lại"

Thay vì bắt decoder nhớ mọi thứ qua một vector, ta giữ lại hidden state của
encoder ở **mọi vị trí** $h_1, \dots, h_T$, và ở **mỗi bước giải mã** $t$ cho
decoder tính lại một bản tóm tắt *có trọng tâm* riêng:

$$e_{t,i} = v^\top \tanh(W_h h_i + W_s s_{t-1}) \qquad \text{(điểm liên quan)}$$

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})} \qquad \text{(softmax → trọng số, tổng = 1)}$$

$$c_t = \sum_{i=1}^{T} \alpha_{t,i}\, h_i \qquad \text{(context vector riêng cho bước } t\text{)}$$

Ý nghĩa từng thành phần:

- $s_{t-1}$ — hidden state decoder bước trước: "tôi đang định nói gì tiếp".
- $W_h, W_s$ — hai phép chiếu học được, đưa hidden encoder và decoder về cùng
  một không gian để so sánh được với nhau.
- $v^\top \tanh(\cdot)$ — một MLP nhỏ 1 lớp ẩn, nén vector so sánh về 1 con số
  điểm. Vì điểm được tính bằng phép *cộng* hai phép chiếu rồi qua MLP, biến thể
  này gọi là **additive attention**. (Luong 2015 đề xuất **multiplicative**:
  $e_{t,i} = s_{t-1}^\top W h_i$ — rẻ hơn, xem bài tập.)
- $\alpha_{t,i}$ — decoder đang "nhìn" vào từ nguồn thứ $i$ nhiều cỡ nào khi sinh
  từ đích thứ $t$. Đây chính là ma trận trong `attention_heatmap.png`: với cặp
  Anh-Việt trật tự từ khá giống nhau, heatmap hiện lên một đường chéo mờ.

Trong code, đầu vào GRU decoder mỗi bước là `[embedding(y_trước) ; c_t]` — từ
vừa sinh nối với bản tóm tắt mới tính.

Attention chính là **tiền thân trực tiếp của Transformer**: "Attention is All
You Need" (Vaswani 2017) giữ lại cơ chế này, bỏ hẳn RNN, và tính attention song
song mọi cặp vị trí (self-attention).

## 4. Teacher forcing và exposure bias

Lúc train, đầu vào bước $t$ của decoder có 2 lựa chọn:

- **Teacher forcing** — đưa từ đích *thật* $y_{t-1}$: hội tụ nhanh, vì model
  đoán sai một từ thì bước sau vẫn được học trên tiền tố đúng (lỗi không dồn chuỗi).
- **Free running** — đưa từ model *vừa đoán*: giống điều kiện lúc dịch thật.

Train 100% teacher forcing tạo **exposure bias**: lúc suy luận model phải tự ăn
output của chính mình — điều nó chưa từng trải qua, nên một lỗi nhỏ có thể kéo
sập cả câu. Script dung hoà bằng `TF_RATIO = 0.5`: mỗi bước tung đồng xu chọn
một trong hai chế độ.

## 5. Suy luận: greedy decoding

Bắt đầu từ `<sos>`, mỗi bước lấy từ có xác suất cao nhất (argmax), dừng khi gặp
`<eos>` hoặc chạm trần 20 từ. Greedy **tham lam**: chọn sai một từ là không quay
lại được, dù tổng thể có chuỗi tốt hơn. **Beam search** giữ $k$ ứng viên tốt
nhất mỗi bước sẽ cho bản dịch tốt hơn với chi phí gấp $k$ lần (bài tập 2).

## 6. Đánh giá bằng BLEU

BLEU-4 (Papineni 2002) so bản dịch máy với bản dịch tham chiếu qua **modified
n-gram precision** với $n = 1..4$:

$$\text{BLEU} = \text{BP} \cdot \exp\!\Big(\frac{1}{4}\sum_{n=1}^{4} \log p_n\Big)$$

- $p_n$ — tỉ lệ n-gram của bản máy có mặt trong tham chiếu, **có clip**: mỗi
  n-gram chỉ được tính tối đa bằng số lần nó xuất hiện trong tham chiếu, chặn
  kiểu ăn gian lặp từ ("the the the").
- $\text{BP} = \min\big(1, e^{1 - \text{len}_{ref}/\text{len}_{hyp}}\big)$ —
  brevity penalty phạt bản dịch *ngắn*, vì precision thuần không phạt dịch thiếu
  (dịch đúng 2 từ rồi dừng → precision 100%!).
- Tính **corpus-level**: cộng dồn số đếm trên toàn tập rồi mới lấy tỉ lệ.

Hạn chế đáng nhớ: BLEU chỉ so *bề mặt chuỗi ký tự* — "tôi thích chó" vs "mình
mê cún" là 0 điểm dù đồng nghĩa; kết quả nhạy với cách tách token; và với tiếng
Việt (viết tách *âm tiết*, không phải *từ*) BLEU word-level thực chất là
syllable-level.

## 7. Dữ liệu và các quyết định tiền xử lý

Corpus: 12.628 cặp câu Anh-Việt từ [Tatoeba](https://tatoeba.org) (đóng gói bởi
manythings.org/anki), script tự tải về `data/tatoeba_envi/`. Sau khi lọc câu
≤ 12 token còn **11.080 cặp** (9.991 train / 906 câu val).

Các quyết định quan trọng (đều có lý do trong comment của script):

| Quyết định | Lý do |
|---|---|
| Chuẩn hoá Unicode **NFC** | "ể" có thể được mã hoá 2 cách; không chuẩn hoá → từ điển tách đôi một từ |
| Tách dấu câu, *không* tách `'` | "chạy!" → 2 token; giữ nguyên "don't", "i'm" |
| Vocab xây từ **train only** (EN 2.431, VI 1.798) | đếm cả val = rò rỉ thông tin tập đánh giá |
| Từ hiếm (< 2 lần) → `<unk>` | vocab gọn, model biết cách xử lý từ lạ |
| Chia train/val **theo câu nguồn** | một câu EN có nhiều bản dịch VI; chia theo cặp thì câu val đã bị "thấy" lúc train → BLEU ảo |
| Encoder dùng `pack_padded_sequence` | GRU không "đọc" các ô `<pad>`; hidden cuối đúng là tại token thật cuối |
| Attention mask vị trí pad bằng $-\infty$ | sau softmax trọng số pad = 0 tuyệt đối |

## 8. Chạy code

```bash
source .venv/bin/activate
python day6/01_seq2seq_attention_translation.py
```

Kết quả một lần chạy thật (Mac, thiết bị MPS, ~7s/epoch, tổng ~3 phút):

```
Cặp câu sau lọc: 11080 (train 9991, val 906)
Vocab EN: 2431 | Vocab VI: 1798
Số tham số: 2,662,150
Epoch  1/15 | loss train 4.845 | loss val 4.136
Epoch 10/15 | loss train 1.442 | loss val 2.522   ← val chạm đáy
Epoch 15/15 | loss train 0.760 | loss val 2.612   ← bắt đầu overfit nhẹ

BLEU-4 trên 906 câu val: 21.0
```

(Số liệu dao động nhẹ giữa các lần chạy — phép toán trên MPS không hoàn toàn
deterministic dù đã cố định seed.)

Loss val chạm đáy quanh epoch 10 rồi nhích lên trong khi loss train tiếp tục
giảm — dấu hiệu **overfitting** kinh điển trên dataset nhỏ; đây là lý do người
ta dùng early stopping (day 3).

Ví dụ dịch từ tập val (câu model chưa từng thấy):

```
EN : i can't buy it because i have no money .
REF: tôi không thể mua nó vì tôi không có tiền .
MÁY: tôi không thể mua nó vì không có tiền .

EN : i don't want anybody to know that i'm rich .
REF: tôi không muốn bất cứ ai biết là tôi giàu .
MÁY: tôi không muốn bất cứ ai biết là tôi giàu .
```

Hai file hình sinh ra cạnh script:

- `seq2seq_loss.png` — loss train/val theo epoch.
- `attention_heatmap.png` — 4 câu val với ma trận $\alpha$: trục x là token
  nguồn EN, trục y là token VI sinh ra; thấy rõ đường chéo alignment và ô
  đậm tại các cặp từ tương ứng.

## 9. Bài tập về nhà

1. **Encoder 2 chiều** — `nn.GRU(..., bidirectional=True)` đọc câu nguồn cả hai
   chiều; cần chiếu `2*D_HID → D_HID` trước khi đưa vào decoder. Đây chính là
   thiết kế gốc trong paper Bahdanau. BLEU thay đổi thế nào?
2. **Beam search** với $k = 3$ thay cho greedy. So sánh BLEU và thời gian dịch.
3. **Luong attention** — thay công thức điểm bằng $e_{t,i} = s_{t-1}^\top W h_i$.
   Đếm số tham số tiết kiệm được và so BLEU.
4. **Dịch ngược Việt→Anh** — chỉ cần đổi chiều cặp câu. Chiều nào BLEU cao hơn?
   Tại sao? (gợi ý: hình thái từ, phía nào nhiều cách diễn đạt hơn)
5. **Câu dài** — tăng `MAX_LEN` lên 20–30, quan sát chất lượng dịch câu dài và
   heatmap của chúng. Điểm nghẽn nào của RNN attention lộ ra?
6. **BLEU đa tham chiếu** — Tatoeba có nhiều bản dịch cho một câu; sửa
   `bleu_corpus` nhận nhiều tham chiếu (clip theo max đếm giữa các tham chiếu,
   BP theo tham chiếu gần độ dài nhất).

## 10. Tài liệu

- Sutskever, Vinyals, Le (2014). *Sequence to Sequence Learning with Neural
  Networks*. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- Bahdanau, Cho, Bengio (2015). *Neural Machine Translation by Jointly Learning
  to Align and Translate*. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Luong, Pham, Manning (2015). *Effective Approaches to Attention-based NMT*.
  [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine
  Translation*. [ACL](https://aclanthology.org/P02-1040/)
- Vaswani et al. (2017). *Attention is All You Need*.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
