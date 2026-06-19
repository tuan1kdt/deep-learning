# MedVQA Desktop Demo — Thiết kế app Go + Wails

**Ngày:** 2026-06-19
**Trạng thái:** Đã duyệt qua brainstorming (3 điều chỉnh: React + frontend-design,
cho đổi checkpoint, đáp án = top-1 dự đoán).
**Phạm vi:** Đồ án giữa kỳ — app demo cho model MedVQA trong `midterm/`.
**Nền tảng liên quan:** [`2026-06-11-medvqa-design.md`](2026-06-11-medvqa-design.md) (model gốc).

---

## 1. Mục tiêu

Một **desktop app** (Go + Wails) để demo trực tiếp model MedVQA giữa kỳ. Người dùng:

1. Upload một ảnh y khoa.
2. Nhập một câu hỏi tiếng Anh.
3. Bấm **Run** → app hiển thị **đáp án** (top-1 dự đoán) + danh sách **top-5** kèm
   xác suất.
4. Nếu checkpoint đang dùng là `cross_attention` → hiển thị **thêm ảnh heatmap
   attention** chồng lên ảnh gốc (model "nhìn" vào vùng nào).
5. Có thể **đổi checkpoint** đang dùng ngay trong UI (concat / hadamard /
   cross_attention…).

### 1.1 Tiêu chí thành công

- **Demo mượt khi trình bày**: model load **một lần**, mỗi câu hỏi trả lời <1s.
- **Tái dùng tối đa code Python sẵn có** — không nhân đôi logic model.
- **UI chỉn chu** (dùng skill `frontend-design`), dễ thao tác trước giám khảo.
- **Chạy local** trên máy Mac của người trình bày (không cần đóng gói phân phối).

### 1.2 Phạm vi — cố tình KHÔNG làm (YAGNI)

- **Không** đóng gói `.app` phân phối (không bưng Python/torch/checkpoint vào bản
  build). Chạy qua `wails dev` hoặc build chạy tại chỗ trên máy có sẵn `.venv`.
- **Không** lưu lịch sử hỏi-đáp, không multi-image, không batch inference.
- **Không** train/đánh giá trong app — app chỉ inference.

---

## 2. Ràng buộc đã chốt

- **Bridge Go↔Python = sidecar HTTP** (đã chọn ở brainstorming). Lý do: model load
  ~15–30s; sidecar giữ model nóng → demo nhiều câu liên tiếp mượt; dễ debug bằng
  `curl`; tái dùng code sẵn có với ~ vài chục dòng Python mới.
- **Chạy local, tái dùng `.venv` sẵn có** (`<repo>/.venv/bin/python`). Không bundle.
- **Frontend = React** (template Wails `react-ts`), dựng bằng skill `frontend-design`.
- **Heatmap = attention thật** từ checkpoint `cross_attention` — đúng logic `demo.py`
  (49 vùng 7×7 → `np.kron` 32× → overlay jet alpha 0.4 trên ảnh 224×224).
- **Đáp án "đúng" = top-1 dự đoán của model** (ảnh upload tùy ý, không có
  ground-truth để so).
- Toolchain đã có: Go 1.26.1, Wails, Node 22, npm 10, Python 3.14 trong `.venv`.

---

## 3. Kiến trúc tổng thể

```
┌──────────────────────────────────────────────────────────┐
│  WAILS APP  (1 cửa sổ desktop, chạy local)                 │
│                                                            │
│  Frontend React               Go backend (app.go)          │
│  • Upload ảnh + preview        • startup: spawn sidecar     │
│  • Ô nhập câu hỏi              • Health() → poll tới ready  │
│  • Dropdown checkpoint         • Checkpoints() → list .pt   │
│  • Nút Run                     • LoadCheckpoint(name)       │
│  • Top-1 + top-5 bars          • Predict(imgB64, question)  │
│  • Ảnh heatmap (nếu có)        • shutdown: kill sidecar     │
│         └──── Wails bindings (Go↔JS) ────┘                  │
└───────────────────────────┬────────────────────────────────┘
                            │ HTTP localhost (cổng tự do do Go cấp)
                            ▼
┌──────────────────────────────────────────────────────────┐
│  SIDECAR PYTHON  (midterm/serve.py, Flask)                 │
│  • GET  /health      → {ready, checkpoint, has_attention}  │
│  • GET  /checkpoints → {checkpoints:[...], current}        │
│  • POST /load {checkpoint}  → reload model (blocking)      │
│  • POST /predict (ảnh + câu hỏi)                           │
│        → {answers:[{answer,prob}×5], heatmap_b64|null,     │
│           has_attention}                                   │
│  • Tái dùng load_model / build_transforms / load_vocab     │
│    + logic heatmap của demo.py                             │
└──────────────────────────────────────────────────────────┘
```

**Vị trí code:**
- `midterm/serve.py` — sidecar (file mới, ~80–100 dòng).
- `midterm/desktop/` — dự án Wails (Go + React).
- `requirements.txt` — thêm `flask`.

**Cách Go định vị repo & venv:** Go đi ngược từ thư mục làm việc lên trên tìm marker
(`.venv` + thư mục `midterm`) để xác định **repo root**; cho phép override bằng biến
môi trường `DEEPLEARNING_ROOT`. Sidecar được spawn bằng `<root>/.venv/bin/python -m
midterm.serve --port <p> --checkpoint <name>` với `cwd = <root>` để `python -m
midterm.serve` resolve được package.

---

## 4. Sidecar Python — `midterm/serve.py`

Flask app (single client local → đủ). **Tái dùng** từ code sẵn có:
`midterm.evaluate.load_model`, `midterm.data.dataset.build_transforms`,
`midterm.data.vocab.load_vocab`, `midterm.config.pick_device`, và đúng đoạn dựng
heatmap trong `midterm/demo.py`.

### 4.1 Trạng thái & vòng đời model

- State toàn cục: `{ready: bool, checkpoint: str|None, has_attention: bool}` + 1 `Lock`.
- **Khởi động**: Flask bind cổng ngay (ready=False), một **thread nền** load
  checkpoint mặc định, xong thì set ready=True. → Go có thể poll `/health` ngay
  trong lúc model đang load.
- **Checkpoint mặc định (robust)**: dùng `--checkpoint` nếu được truyền và file tồn
  tại; ngược lại ưu tiên `cross_attention.pt` nếu có; nếu vẫn không có thì lấy file
  `*.pt` đầu tiên trong `checkpoint_dir`. *Vì sao?* Hiện local mới chỉ có `concat.pt`
  (chưa có `cross_attention.pt`) → mặc định cứng vào cross_attention sẽ fail lúc khởi
  động trong giai đoạn phát triển. Fallback giúp app luôn lên được với checkpoint
  bất kỳ; heatmap chỉ tắt đi khi đó không phải cross_attention.
- **`has_attention`** = True khi `cfg.fusion == "cross_attention"` (đọc từ config
  lưu trong checkpoint sau khi `load_model`).

### 4.2 Endpoints

| Method | Path | Vào | Ra |
|---|---|---|---|
| GET | `/health` | — | `{ready, checkpoint, has_attention}` |
| GET | `/checkpoints` | — | `{checkpoints:[tên .pt trong checkpoint_dir], current}` |
| POST | `/load` | `{checkpoint: "<tên>"}` | **blocking** tới khi load xong → `{ready:true, checkpoint, has_attention}`; lỗi → 4xx + `{error}` |
| POST | `/predict` | multipart: `image` (file) + `question` (text) | `{answers:[{answer,prob}×5], heatmap, has_attention}` |

- **`/predict`** dùng **multipart** (đơn giản nhất để gửi file từ Go). Trả `heatmap`
  = chuỗi PNG base64 khi `has_attention=True`, ngược lại `null`. Heatmap dựng vào
  buffer `BytesIO` rồi base64 — **không** ghi file tạm.
- Trong lúc đang `/load` (ready=False): `/predict` trả `409 {error:"model loading"}`.
- `/checkpoints` liệt kê mọi file `*.pt` trong `cfg.checkpoint_dir`, tên hiển thị =
  stem (vd `cross_attention`).

### 4.3 Định dạng đáp án

```json
{
  "answers": [{"answer": "no", "prob": 0.81}, {"answer": "yes", "prob": 0.12}, ...],
  "heatmap": "<base64 png>" | null,
  "has_attention": true
}
```

---

## 5. Go backend — `midterm/desktop/app.go`, `main.go`

### 5.1 Vòng đời sidecar

- `startup(ctx)`:
  1. Resolve repo root + `.venv/bin/python` + checkpoint mặc định.
  2. Tìm **cổng TCP tự do** (`net.Listen("tcp", ":0")` rồi đóng, lấy port).
  3. `exec.Command(venvPython, "-m", "midterm.serve", "--port", port, "--checkpoint",
     default)` với `Dir = root`; lưu handle; gom stderr để báo lỗi.
- `shutdown(ctx)`: kill process sidecar (tránh tiến trình mồ côi).

### 5.2 Method bind cho frontend

| Method Go | Vai trò |
|---|---|
| `Health() (HealthResp, error)` | proxy `GET /health` — frontend poll lúc khởi động & sau khi đổi checkpoint |
| `Checkpoints() (CheckpointsResp, error)` | proxy `GET /checkpoints` — đổ vào dropdown |
| `LoadCheckpoint(name string) (HealthResp, error)` | proxy `POST /load` (blocking) — đổi checkpoint |
| `Predict(imageB64, question string) (PredictResp, error)` | decode base64 → POST multipart `/predict` |

- Go là proxy mỏng: chỉ chuyển tiếp HTTP nội bộ ra binding để frontend không gọi
  HTTP trực tiếp (giữ một điểm vào, dễ xử lý lỗi).

---

## 6. Frontend React — `midterm/desktop/frontend/`

Dựng bằng skill **`frontend-design`** để UI đẹp & nhất quán. Một màn hình, 2 cột.

```
┌───────────────────────────────────────────────────────┐
│  MedVQA Demo — VQA-RAD     Checkpoint:[cross_attention▾]│
│                                       ● Model: sẵn sàng │
├──────────────────────────┬────────────────────────────┤
│  [ Kéo-thả / Chọn ảnh ]   │   Đáp án:                   │
│  ┌────────────────────┐  │   ┌──────────────────────┐  │
│  │   (preview ảnh)    │  │   │       no             │  │ ← top-1 chữ to
│  └────────────────────┘  │   └──────────────────────┘  │
│  Câu hỏi:                │   Top-5:                    │
│  ┌────────────────────┐  │   no   ▓▓▓▓▓▓▓▓ 0.81        │
│  │ is there ...?      │  │   yes  ▓▓ 0.12              │
│  └────────────────────┘  │   ...                       │
│        [  Run  ]         │   Heatmap (nếu cross_attn):  │
│                          │   ┌──────────────────────┐  │
│                          │   │  (overlay attention) │  │
│                          │   └──────────────────────┘  │
└──────────────────────────┴────────────────────────────┘
```

- **Nhãn UI tiếng Việt**; câu hỏi nhập **tiếng Anh** (model tiếng Anh).
- Ảnh chọn bằng `<input type=file>` (+ kéo-thả), đọc thành base64 (FileReader) gửi
  qua `Predict`. Preview hiển thị ngay.
- **Dropdown checkpoint** đổ từ `Checkpoints()`; đổi → gọi `LoadCheckpoint(name)` →
  overlay "Đang tải model…" cho tới khi xong.
- Panel **heatmap** chỉ hiện khi `has_attention=true`; checkpoint khác → ẩn kèm ghi
  chú "Checkpoint này không có attention để vẽ heatmap".
- Heatmap nhận base64 → hiển thị bằng `<img src="data:image/png;base64,...">`.

---

## 7. Luồng dữ liệu

**Khởi động**
1. Mở app → Go spawn sidecar (cổng tự do) → frontend hiện "Đang tải model… (~20s)".
2. Frontend poll `Health()` mỗi 1s → khi `ready=true`: bật Run, gọi `Checkpoints()`
   đổ dropdown, set trạng thái `has_attention`.

**Hỏi-đáp**
3. User chọn ảnh (preview) + gõ câu hỏi → **Run**.
4. `Predict(imgB64, q)` → Go POST multipart `/predict` → sidecar inference.
5. Sidecar trả top-5 (+ heatmap nếu cross_attention) → frontend hiển thị.

**Đổi checkpoint**
6. User chọn checkpoint khác → `LoadCheckpoint(name)` (blocking, overlay "đang tải")
   → xong: cập nhật `has_attention`, ẩn/hiện panel heatmap, xóa kết quả cũ.

---

## 8. Xử lý lỗi

| Tình huống | Xử lý |
|---|---|
| Thiếu `.venv` / `python` không chạy được | Go bắt lỗi spawn → frontend báo "Không khởi động được sidecar" + stderr |
| Thiếu/sai checkpoint | sidecar trả lỗi load → frontend báo, dropdown vẫn dùng được để chọn cái khác |
| Health quá lâu (>90s) | frontend báo timeout, gợi ý kiểm tra log |
| Thiếu ảnh hoặc câu hỏi | nút Run bị khóa tới khi đủ cả hai |
| `/predict` lỗi (ảnh hỏng, exception) | hiện thông báo lỗi, app vẫn dùng tiếp |
| Đang load mà bấm Run | nút Run khóa khi `ready=false`; sidecar cũng trả 409 phòng hờ |
| Checkpoint không phải cross_attention | vẫn hiện đáp án; ẩn panel heatmap + ghi chú |

---

## 9. Kiểm thử

- **Sidecar (tự động, pytest)** — phần logic chính, test độc lập GUI:
  - `GET /health` trả cấu trúc đúng; `ready` chuyển True sau khi load.
  - `GET /checkpoints` liệt kê đúng các `.pt` có trong `checkpoint_dir`.
  - `POST /predict` với 1 ảnh mẫu VQA-RAD + câu hỏi → trả 5 đáp án, mỗi cái có
    `answer`+`prob`; với checkpoint cross_attention → `heatmap` khác null.
  - `POST /load` đổi sang checkpoint khác → `has_attention` cập nhật đúng.
- **Go binding** — test `Predict()`/`Health()` với mock HTTP server (hoặc sidecar
  thật) → xác nhận parse JSON & lỗi đúng.
- **Thủ công (`wails dev`)** — load ảnh X-quang ngực, hỏi "is there cardiomegaly?"
  với checkpoint cross_attention → kiểm tra đáp án + heatmap; đổi sang concat →
  heatmap ẩn, vẫn ra đáp án.

---

## 10. Các đơn vị & ranh giới

| Đơn vị | Làm gì | Phụ thuộc | Test thế nào |
|---|---|---|---|
| `serve.py` | Inference qua HTTP, giữ model nóng | code model sẵn có, Flask | pytest gọi endpoint |
| `app.go` | Vòng đời sidecar + proxy bindings | `serve.py` qua HTTP | mock HTTP |
| Frontend React | Thu thập input, hiển thị kết quả/heatmap | bindings Go | thủ công |

Ba đơn vị giao tiếp qua interface rõ ràng (HTTP JSON, Wails bindings), thay đổi nội
bộ từng cái không phá cái kia.

---

## 11. Thứ tự triển khai (tóm tắt cho bước writing-plans)

1. `serve.py` + thêm `flask` vào requirements; test bằng `curl`/pytest **trước**.
2. Khởi tạo dự án Wails React ở `midterm/desktop/`; `app.go` spawn + bindings.
3. Frontend React (skill `frontend-design`): upload, câu hỏi, Run, kết quả, heatmap,
   dropdown checkpoint, trạng thái loading.
4. Ghép nối + kiểm thử thủ công end-to-end với checkpoint cross_attention.
