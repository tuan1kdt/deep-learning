# ViVLM-nano — VLM tiếng Việt tự pretrain từ đầu

Bài tự học sau đồ án cuối kỳ (`final/`), đi phần còn trống trong sơ đồ kiến
trúc multimodal của giảng viên: **tự pretrain một LLM từ đầu** (không chỉ
fine-tune) rồi mới ghép fusion ảnh-chữ. Pipeline trọn vòng đời LLM hiện đại
trên 1 GPU 24GB:

**pretrain GPT tiếng Việt (~100M tham số, ~3B token) → fusion SigLIP (LLaVA-style)
→ SFT hai pha → SCST (RL, tối ưu trực tiếp CIDEr)**

Thiết kế chi tiết ban đầu: `docs/superpowers/specs/2026-07-18-vivlm-nano-design.md`
(`docs/` không commit — xem ghi chú trong lịch sử làm việc).

## 1. Lý thuyết

### 1.1 Vì sao pretrain rồi mới SFT?

Ở `final/`, bài toán captioning được đặt thẳng là sinh câu **có điều kiện
ảnh**: decoder học từ đầu trên vài nghìn cặp ảnh-caption, không có giai đoạn
nào học ngôn ngữ tự do trước. ViVLM-nano tách bài toán ra làm ba giai đoạn,
đúng công thức mà GPT/Llama/LLaVA hiện đại dùng, để trả lời câu hỏi: khả năng
"biết tiếng Việt" và khả năng "mô tả được ảnh" có phải học chung một lúc hay
tách được?

Giai đoạn A (pretrain) huấn luyện next-token prediction thuần túy trên ~3 tỷ
token văn bản tiếng Việt (không ảnh). Đây là dạng self-supervised rẻ nhất:
nhãn chính là token kế tiếp, dữ liệu là văn bản thô không giới hạn. Mục tiêu
không phải "học nói tiếng Việt hay" mà là học một **phân phối xác suất
P(token kế | ngữ cảnh)** đủ tốt để làm nền — biểu diễn cú pháp, ngữ nghĩa,
world knowledge nằm trong trọng số trước khi model nhìn thấy ảnh nào.

Giai đoạn B (SFT) không dạy ngôn ngữ từ đầu nữa mà chỉ **nắn format**: cho
model học đối chiếu 49 token ảnh (từ SigLIP) với một câu trả lời cụ thể, học
theo prompt template `<|user|> ... <|assistant|> ...`. Vì nền ngôn ngữ đã có
sẵn từ giai đoạn A, SFT hội tụ nhanh (vài giờ, vài chục nghìn mẫu) thay vì cần
hàng triệu cặp ảnh-câu như học từ đầu.

Giai đoạn C (SCST/RL) sửa một lệch pha kinh điển của SFT: loss huấn luyện là
cross-entropy token-level, nhưng thước đo cuối cùng ta quan tâm là CIDEr —
một hàm không khả vi trên toàn câu (so khớp n-gram với nhiều câu tham chiếu).
Cross-entropy tối ưu "đúng từng từ giống ground-truth", không tối ưu trực
tiếp "câu sinh ra được đánh giá tốt". REINFORCE cho phép tối ưu thẳng một
reward không khả vi bằng cách coi việc sample câu là một hành động, nhân xác
suất log của hành động đó với reward nó nhận được.

So trực tiếp với `final/`: captioning ở đó là generation có điều kiện ảnh,
gói gọn trong một mô hình, một giai đoạn train. Ở đây, captioning chỉ là
**một ứng dụng cuối** của một LLM tổng quát — cùng kiến trúc decoder-only,
nhưng quy mô dữ liệu và số giai đoạn train phản ánh cách các LLM thật (GPT,
Llama, Qwen-VL, LLaVA...) được xây: bỏ điều kiện ảnh ở giai đoạn đầu, scale
dữ liệu text tối đa, rồi mới "dạy nhìn" bằng một lượng dữ liệu nhỏ hơn nhiều.

### 1.2 Kiến trúc Llama-style vs GPT-2

`vivlm/models/gpt.py` là decoder-only Transformer nhưng thay 3 thành phần
của GPT-2 nguyên bản bằng phiên bản Llama-style (chuẩn thực hành từ 2023 trở
đi — Llama, Mistral, Qwen đều dùng):

| Thành phần | GPT-2 | ViVLM-nano (Llama-style) | Vì sao tốt hơn |
|---|---|---|---|
| Vị trí | Learned positional embedding (bảng tra theo index tuyệt đối) | RoPE (Rotary Position Embedding) — xoay từng cặp chiều của q/k một góc tỷ lệ vị trí | RoPE mã hóa vị trí **tương đối** ngay trong tích vô hướng q·k (không cần cộng thêm vector vị trí vào embedding), nên attention score giữa hai token phụ thuộc khoảng cách chứ không phụ thuộc vị trí tuyệt đối — ngoại suy tốt hơn ra ngoài độ dài đã train, không tốn tham số bảng tra |
| Chuẩn hóa | LayerNorm (trừ mean, chia std, có bias) | RMSNorm — chỉ chia căn trung bình bình phương, không trừ mean, không bias | Bỏ bước center hóa mà thực nghiệm cho thấy không cần thiết để ổn định gradient; rẻ hơn ~10-15% FLOP cho norm, ít tham số hơn (không bias, không mean) |
| MLP | 2 lớp Linear + GELU | SwiGLU: `w2(silu(w1(x)) * w3(x))` — 3 ma trận, gate nhân điểm | Cơ chế gating (nhân hai nhánh linear trước khi qua down-projection) cho chất lượng/FLOP tốt hơn GELU-MLP ở cùng số tham số hiệu dụng (kết quả thực nghiệm trong paper GLU Variants, Shazeer 2020) — đánh đổi bằng việc có thêm 1 ma trận nên `mlp_hidden` phải giảm (~8/3·d_model thay vì 4·d_model) để giữ số tham số tương đương |

Ngoài ra: không bias ở mọi Linear (kể cả attention/MLP), weight tying giữa
`tok_emb` và `lm_head`, và khởi tạo residual projection với
`std=0.02/sqrt(2·n_layer)` (theo GPT-2 paper) để tổng phương sai residual
không bùng nổ theo độ sâu.

### 1.3 Fusion kiểu LLaVA

Ảnh được nối thẳng vào chuỗi token văn bản làm **prefix**, thay vì có một
khối cross-attention riêng biệt (khác `final/`, nơi decoder Transformer tự
cài để lộ cross-attention weight cho heatmap). Sơ đồ:

```
ảnh 224x224
   │
   ▼
SigLIP-B/16 (frozen, from_pretrained, requires_grad_(False))
   │  last_hidden_state: 196 patch token × 768 dim  (grid 14x14)
   ▼
pixel-shuffle (gộp mỗi ô 2x2 patch lân cận thành 1 token,
               768 dim -> 4*768=3072 dim, 196 -> 49 token)
   │
   ▼
MLP projector: Linear(3072->768) -> SiLU -> Linear(768->768)
   │  49 token ảnh, cùng d_model=768 với GPT
   ▼
prepend  ┌────────────────────────────────────────────┐
────────►│ [img_0 .. img_48]  [<|user|> ... <|assistant|> ...] │
         └────────────────────────────────────────────┘
                     │
                     ▼
        GPT Llama-style (self-attention causal + RoPE, 12 lớp)
        — token ảnh và token chữ nằm CHUNG một chuỗi, attention
          score giữa chúng học được từ dữ liệu, không cần layer
          cross-attention riêng
                     │
                     ▼
        loss chỉ tính trên phần response (49 vị trí ảnh + prompt
        bị mask -100 trong labels)
```

Vì sao không cần cross-attention riêng như `final/`: ở `final/`, decoder chưa
từng thấy ảnh trong pretrain nên phải có một cơ chế tường minh "mỗi bước sinh
từ, nhìn vùng ảnh nào" (Bahdanau attention / cross-attention Transformer) để
mô hình học liên kết ảnh-chữ từ đầu, trên rất ít dữ liệu (8k ảnh). Ở đây, GPT
đã có sẵn self-attention full (causal) mạnh từ pretrain; khi ảnh được đưa vào
làm token đầu chuỗi, cơ chế self-attention *đã có sẵn* sẽ tự học attend vào
đúng token ảnh cần thiết khi sinh từng từ — không cần thêm tham số hay cấu
trúc mới, miễn là SFT đủ dữ liệu để "dạy" self-attention làm việc đó (đây
chính là ý tưởng của LLaVA: fusion rẻ nhất có thể, tận dụng năng lực attention
đã pretrain). Đánh đổi: mất khả năng trực quan hóa "từ nào nhìn vùng ảnh nào"
một cách tách bạch như heatmap cross-attention của `final/` — muốn xem phải
lấy attention weight ở các layer self-attention, khó diễn giải hơn.

### 1.4 SCST

Self-Critical Sequence Training (Rennie et al., 2017) là REINFORCE với
baseline là chính model decode greedy — không cần train thêm một value
network như actor-critic:

```
r_greedy = CIDEr(caption sinh bằng greedy decode, refs)
r_sample = CIDEr(caption sinh bằng sampling ngẫu nhiên, refs)
advantage = r_sample - r_greedy

loss = -advantage * Σ_t log P(token_sample_t | ảnh, prompt, token_sample_<t)
```

(`vivlm/scst.py::scst_loss`, tính trung bình theo batch). Nếu câu sample được
CIDEr cao hơn câu greedy, advantage dương → gradient đẩy xác suất của đúng
chuỗi đã sample lên; nếu tệ hơn, advantage âm → đẩy xuống. Baseline greedy
làm giảm variance của gradient estimator so với REINFORCE trần (baseline = 0
hoặc baseline = trung bình reward batch): vì greedy và sample dùng cùng một
ảnh, cùng một model tại cùng một step, phần "khó" chung của mẫu đó (ảnh mờ,
caption đa nghĩa) bị trừ đi ở cả hai vế, chỉ còn lại tín hiệu "sample này tốt
hơn/tệ hơn cách model *tự tin nhất* sẽ trả lời" — đúng bản chất một baseline
tốt trong policy gradient.

Liên hệ `final/`: đồ án cuối kỳ dừng ở MLE (cross-entropy) thuần, không có
giai đoạn RL — SCST ở đây là phần mở rộng trực tiếp mà `final/` liệt vào
hướng phát triển nhưng chưa làm, tối ưu thẳng CIDEr thay vì proxy
cross-entropy.

## 2. Chạy trên Mac (smoke)

```bash
source .venv/bin/activate
pytest vivlm/tests -v
```

`pretrain.py` không có flag `--train-bin/--val-bin` (đọc đường dẫn mặc định
trong `PretrainConfig`), nên để smoke thủ công trên Mac cần tự tạo hai file
`.bin` nhỏ trước — đúng cách `vivlm/tests/test_pretrain.py` làm (token ngẫu
nhiên, không cần tokenizer thật, chỉ để kiểm tra vòng lặp train/log/checkpoint
chạy được):

```bash
python - <<'EOF'
import numpy as np, os
os.makedirs("vivlm/data/bin", exist_ok=True)
np.random.randint(0, 20480, 5000, dtype=np.uint16).tofile("vivlm/data/bin/train.bin")
np.random.randint(0, 20480, 1000, dtype=np.uint16).tofile("vivlm/data/bin/val.bin")
EOF
python -m vivlm.pretrain --max-steps 3 --micro-batch 2 --device cpu
```

`--device cpu` bỏ qua `torch.compile` (chỉ kích hoạt khi `device_type=="cuda"`
trong `train()`), nên không cần `--no-compile` khi chạy trên Mac/CPU/MPS.

## 3. Runbook trainbox (theo thứ tự)

Trainbox: xem `[[trainbox-remote-setup]]` — alias SSH `trainbox`, WSL2 Ubuntu
trên Windows, GPU RTX PRO 4000 Blackwell 24GB, venv `~/work/.venv` (torch
cu128 — Blackwell bắt buộc cu128+). `uv` có sẵn ở `~/.local/bin/uv` trên
trainbox; trên Mac `uv` có thể không nằm trong PATH nên các lệnh dưới đều ghi
kèm fallback `pip`.

```bash
# 0. Cài deps (một lần, trên trainbox — qua WSL)
ssh trainbox 'wsl -e bash -lc "source ~/work/.venv/bin/activate && \
  uv pip install tokenizers transformers huggingface_hub pyarrow datasets pillow pycocoevalcap"'
# Fallback nếu uv không có trên PATH (vd. chạy tương đương trên Mac):
# pip install tokenizers transformers huggingface_hub pyarrow datasets pillow pycocoevalcap
#
# LƯU Ý: cài transformers có thể downgrade tokenizers (0.23 -> 0.22) —
# vô hại (suite vẫn xanh), nhưng nếu import lỗi lạ thì pin lại:
#   uv pip install "tokenizers>=0.22,<0.23"

# 1. Đẩy code (bare repo remote `trainbox` đã tạo theo quy trình final)
git push trainbox main

# 2. Tokenizer (~30 phút, tải file parquet đầu ~1.2GB)
#    (chạy trong tmux; mọi lệnh dài sau đây đều vậy — attach qua SSH,
#     KHÔNG để wsl.exe cuối cùng thoát, nếu không VM WSL chết giữa train)
P=$(python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('epfml/FineWeb2-HQ','vie_Latn/000_00000.parquet', \
  repo_type='dataset'))")
python -m vivlm.tokenizer_train --parquet "$P" --out vivlm/data/tokenizer.json
python -m vivlm.tokenizer_train --compare   # bảng fertility cho báo cáo (cần mạng)

# 3. Token hóa 3B token (~1-2h CPU 24 lõi + tải ~10-15GB parquet)
python -m vivlm.data.prepare_pretrain --target-tokens 3e9 --val-tokens 3e7

# 4. Smoke đo tốc độ (10 phút) -> chỉnh micro-batch sát 24GB
python -m vivlm.pretrain --max-steps 30
#    xem tok/s + vram in ra mỗi 10 step; tăng --micro-batch đến khi vram ~22-23GB
#    tok/s * 86400 < 3e9? -> hạ target: --max-steps 4000 (~2B token) đủ trong 1 ngày
#    torch.compile chỉ có tác dụng trên CUDA (train() chỉ compile khi
#    device_type=="cuda") — bước smoke này CŨNG là chỗ xác nhận compile
#    hoạt động trên máy Blackwell/cu128 (không lỗi biên dịch/backend);
#    nếu compile lỗi hoặc làm chậm hơn, chạy lại với --no-compile

# 5. Pretrain thật (~12-14h) — pane 2 chạy: bash vivlm/watch_gpu.sh
python -m vivlm.pretrain --micro-batch <N>
#    đứt giữa chừng, resume-safe (optimizer + RNG lưu trong checkpoint):
#    python -m vivlm.pretrain --micro-batch <N> \
#      --resume vivlm/checkpoints/pretrain/latest.pt

# 6. Kiểm tra định tính + PPL + so baseline (bits/char — PPL token không
#    so được giữa 2 tokenizer khác nhau, vd. so với NlpHUST/gpt2-vietnamese)
python -m vivlm.sample --prompt "Việt Nam là"
python -m vivlm.evaluate --mode ppl --ckpt vivlm/checkpoints/pretrain/latest.pt
python -m vivlm.evaluate --mode bpc --ckpt vivlm/checkpoints/pretrain/latest.pt

# 7. Data SFT (~2GB ảnh: UIT-ViIC + KTVIC + OpenViVQA)
python -m vivlm.data.prepare_sft
#    SANITY CHECK bắt buộc sau bước này: so số dòng in ra (train/val/test_*)
#    với số lượng annotation gốc trên HF (UIT-ViIC/KTVIC train+validation,
#    OpenViVQA train+dev). OpenViVQA index ảnh theo BASENAME khi giải nén
#    zip (`_vivqa_rows` dùng os.walk + fmap[f] = ...) — nếu hai ảnh khác
#    thư mục con trùng basename, một ảnh sẽ bị GHI ĐÈ/collide và annotation
#    trỏ tới nó bị bỏ qua lặng lẽ (`if fname not in fmap: continue`).
#    Đối chiếu: len(rows) trả về có khớp số annotation trong
#    vlsp2023_{train,dev}_data.json không? Lệch đáng kể -> kiểm tra basename
#    trùng lặp trong zip trước khi train SFT.

# 8. SFT hai pha (~5-6h)
python -m vivlm.sft --phase projector
python -m vivlm.sft --phase full

# 9. Eval SFT
python -m vivlm.evaluate --mode caption --ckpt vivlm/checkpoints/sft/full.pt \
  --jsonl vivlm/data/sft/test_viic.jsonl
python -m vivlm.evaluate --mode caption --ckpt vivlm/checkpoints/sft/full.pt \
  --jsonl vivlm/data/sft/test_ktvic.jsonl --beam 3
python -m vivlm.evaluate --mode vqa --ckpt vivlm/checkpoints/sft/full.pt \
  --jsonl vivlm/data/sft/test_vivqa.jsonl

# 10. SCST (~2-3h) + eval lại
python -m vivlm.scst
python -m vivlm.evaluate --mode caption --ckpt vivlm/checkpoints/scst/scst.pt \
  --jsonl vivlm/data/sft/test_ktvic.jsonl

# 11. Vẽ + kéo kết quả về Mac
python -m vivlm.plot_logs --which pretrain --csv vivlm/outputs/pretrain_log.csv \
  --out vivlm/outputs/pretrain_loss.png
python -m vivlm.plot_logs --which sft --csv vivlm/outputs/sft_log.csv \
  --out vivlm/outputs/sft_loss.png
python -m vivlm.plot_logs --which gpu --csv vivlm/outputs/gpu_log.csv \
  --out vivlm/outputs/gpu_util.png
# từ Mac:
rsync -avz trainbox-wsl:~/work/deepLearning/vivlm/outputs/ vivlm/outputs/
rsync -avz trainbox-wsl:~/work/deepLearning/vivlm/checkpoints/ vivlm/checkpoints/
```

## 4. Kết quả (điền sau khi train)

| Chỉ số | Giá trị |
|---|---|
| Fertility ours / PhoGPT / GPT-2 | — |
| Val loss cuối / PPL | — |
| bits/char: ours vs NlpHUST/gpt2-vietnamese | — |
| CIDEr ViIC: SFT greedy / beam3 / +SCST | — |
| CIDEr KTVIC: SFT greedy / beam3 / +SCST | — |
| CIDEr VQA (dev) | — |
| GPU util trung bình / VRAM đỉnh | — |
| Tổng giờ GPU | — |

## 5. Nhật ký & bài học

*(ghi thêm trong lúc chạy thật trên trainbox: phát hiện, lỗi gặp phải, quyết
định đổi hyperparameter, số liệu trung gian...)*

- **Deps trainbox**: cài `tokenizers transformers huggingface_hub pyarrow
  datasets pillow pycocoevalcap` bằng `uv pip install` (uv có sẵn ở
  `~/.local/bin/uv` trên trainbox; trên Mac dùng `pip` nếu `uv` không nằm
  trong PATH). Cài `transformers` có thể tự downgrade `tokenizers` từ 0.23
  xuống 0.22 — vô hại, `pytest vivlm/tests` vẫn xanh sau khi downgrade; chỉ
  cần pin lại (`tokenizers>=0.22,<0.23`) nếu gặp lỗi import cụ thể.
- **`prepare_sft` — sanity check số dòng**: bắt buộc so số dòng mỗi file
  jsonl in ra sau `python -m vivlm.data.prepare_sft` với số annotation gốc
  trên HF. `OpenViVQA` giải nén zip ảnh rồi index theo **basename** (không
  giữ cấu trúc thư mục con) — nếu hai ảnh ở hai thư mục con khác nhau trùng
  tên file, một ảnh bị ghi đè và annotation trỏ tới ảnh biến mất khỏi tập dữ
  liệu mà không có cảnh báo nào (`_vivqa_rows` chỉ `continue` lặng lẽ khi
  basename không có trong map). Verify trước khi tin số liệu SFT.
- **`torch.compile` chỉ có tác dụng trên CUDA**: `train()` trong
  `vivlm/pretrain.py` chỉ gọi `torch.compile(model)` khi
  `device_type == "cuda"` — bước smoke (mục 3.4) phải xác nhận compile chạy
  được (không lỗi biên dịch Triton/backend) trên trainbox Blackwell/cu128
  trước khi vào pretrain thật; nếu compile gây lỗi hoặc chậm hơn không
  compile, dùng `--no-compile`.
