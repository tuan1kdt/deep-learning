# MedVQA Desktop Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Một desktop app (Go + Wails + React) demo model MedVQA: upload ảnh + nhập câu hỏi → đáp án top-5 và (với checkpoint `cross_attention`) ảnh heatmap attention.

**Architecture:** Wails app spawn một sidecar Python (Flask) giữ model PyTorch "nóng" trong RAM và phục vụ inference qua `localhost`. Go là proxy mỏng giữa frontend React và sidecar. Mọi logic model tái dùng code sẵn có trong `midterm/` — không nhân đôi.

**Tech Stack:** Python 3.14 + Flask + PyTorch (qua `.venv` sẵn có), Go 1.26 + Wails v2.12, React + Vite + TypeScript (template `react-ts`).

**Spec:** [`docs/superpowers/specs/2026-06-19-medvqa-desktop-demo-design.md`](../specs/2026-06-19-medvqa-desktop-demo-design.md)

## Global Constraints

- **Chạy local**, không đóng gói phân phối. Sidecar dùng interpreter `<repo>/.venv/bin/python`.
- **Tái dùng** `midterm.evaluate.load_model`, `midterm.data.dataset.build_transforms`, `midterm.data.vocab.load_vocab`, `midterm.config.{Config,pick_device}`. KHÔNG copy lại logic model.
- **Heatmap** dựng đúng như `midterm/demo.py`: `attn (1,49)` → reshape `7×7` → `np.kron(heat, ones((32,32)))` → overlay `cmap="jet", alpha=0.4` trên ảnh resize `224×224`, `dpi=120`.
- **Đáp án = top-1 dự đoán** của model (không có ground-truth cho ảnh upload).
- **Heatmap chỉ khi `cfg.fusion == "cross_attention"`** (khi đó `attn is not None`); checkpoint khác → `heatmap=null`, ẩn panel.
- **Sidecar bind `127.0.0.1`** (chỉ localhost). Cổng do Go cấp (free port), truyền qua `--port`.
- **Wails project ở `midterm/desktop/`**, template `react-ts`.
- Nhãn UI tiếng Việt; câu hỏi model nhập tiếng Anh.

---

## File Structure

| File | Trách nhiệm | Task |
|---|---|---|
| `midterm/serve.py` (mới) | Flask sidecar: vòng đời model + 4 endpoint | 1 |
| `midterm/tests/test_serve.py` (mới) | pytest cho sidecar (fast + integration) | 1 |
| `requirements.txt` (sửa) | thêm `flask` | 1 |
| `midterm/desktop/app.go` (sửa từ template) | spawn/kill sidecar + bindings HTTP proxy | 2 |
| `midterm/desktop/main.go` (sửa từ template) | cấu hình cửa sổ + OnStartup/OnShutdown/Bind | 2 |
| `midterm/desktop/app_test.go` (mới) | Go test bindings với httptest mock | 2 |
| `midterm/desktop/frontend/src/lib.ts` (mới) | helper thuần (strip data URL, format prob) | 3 |
| `midterm/desktop/frontend/src/lib.test.ts` (mới) | vitest cho helper | 3 |
| `midterm/desktop/frontend/src/App.tsx` (sửa) | UI 1 màn hình + state machine | 3 |
| `midterm/desktop/frontend/src/App.css` (sửa) | style (frontend-design) | 3 |
| `midterm/desktop/README.md` (mới) | cách chạy app | 4 |

---

## Task 1: Python sidecar (`midterm/serve.py`) + tests

**Files:**
- Create: `midterm/serve.py`
- Create: `midterm/tests/test_serve.py`
- Modify: `requirements.txt` (thêm `flask>=3.0`)

**Interfaces:**
- Consumes: `load_model(path, device) -> (model, cfg)`; `build_transforms(cfg, train) -> Compose`; `load_vocab(path) -> dict`; `pick_device()`; `Config()` (đọc `checkpoint_dir`, `vocab_path`, `text_model_name`, `max_question_len`, `fusion`); model forward `(logits, attn)` với `attn` là `(1,49)` hoặc `None`.
- Produces (HTTP contract dùng ở Task 2):
  - `GET /health` → `{"ready": bool, "checkpoint": str|null, "has_attention": bool}`
  - `GET /checkpoints` → `{"checkpoints": [str], "current": str|null}`
  - `POST /load` body `{"checkpoint": str}` → `{health...}` (200) | `{"error": str}` (400/404)
  - `POST /predict` multipart `image`(file)+`question`(text) → `{"answers":[{"answer":str,"prob":float}×5], "heatmap": str|null, "has_attention": bool}` (200) | `{"error":str}` (400/409)
- Produces (Python API dùng trong test): `available_checkpoints() -> list[str]`; `resolve_default_checkpoint(preferred: str|None) -> str|None`; `render_heatmap(attn, image) -> str`; `load_checkpoint(name: str) -> None`.

- [ ] **Step 1: Thêm Flask vào requirements và cài**

Sửa `requirements.txt`, thêm dòng:
```
flask>=3.0
```
Run:
```bash
source .venv/bin/activate
pip install "flask>=3.0" pytest
```
Expected: cài thành công flask + pytest.

- [ ] **Step 2: Viết test fast cho các hàm thuần (chưa cần model)**

Create `midterm/tests/test_serve.py`:
```python
"""Test sidecar. Phần fast không cần load model; phần integration (cuối file)
load checkpoint thật nên chậm (~30s) và cần checkpoints/ có ít nhất 1 file .pt."""
import base64

import numpy as np
import pytest
import torch
from PIL import Image

from midterm import serve


def test_available_checkpoints_lists_existing_pt():
    names = serve.available_checkpoints()
    # repo hiện có concat.pt; danh sách là tên stem, không trùng, đã sort
    assert "concat" in names
    assert names == sorted(names)


def test_resolve_default_prefers_existing_preferred():
    names = serve.available_checkpoints()
    assert serve.resolve_default_checkpoint("concat") == "concat"


def test_resolve_default_falls_back_when_preferred_missing():
    # tên không tồn tại → fallback (cross_attention nếu có, ngược lại file đầu)
    result = serve.resolve_default_checkpoint("khong_ton_tai")
    assert result in serve.available_checkpoints()


def test_render_heatmap_returns_png_base64():
    attn = torch.rand(1, 49)
    img = Image.new("RGB", (300, 300), "white")
    b64 = serve.render_heatmap(attn, img)
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # magic bytes PNG


def test_health_endpoint_before_load():
    client = serve.app.test_client()
    body = client.get("/health").get_json()
    assert set(body) == {"ready", "checkpoint", "has_attention"}


def test_checkpoints_endpoint_structure():
    client = serve.app.test_client()
    body = client.get("/checkpoints").get_json()
    assert "concat" in body["checkpoints"]
    assert "current" in body
```

- [ ] **Step 3: Chạy test → fail vì chưa có `serve.py`**

Run:
```bash
source .venv/bin/activate && python -m pytest midterm/tests/test_serve.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'midterm.serve'`.

- [ ] **Step 4: Viết `midterm/serve.py`**

Create `midterm/serve.py`:
```python
"""Sidecar HTTP server cho desktop demo: giữ model MedVQA "nóng" trong RAM và
trả lời inference qua localhost. Tái dùng toàn bộ logic model sẵn có
(load_model / build_transforms / load_vocab) — không nhân đôi.

Chạy: python -m midterm.serve --port 8765 [--checkpoint cross_attention]
"""
import argparse
import base64
import io
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render heatmap ra buffer, không cần GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoTokenizer

from midterm.config import Config, pick_device
from midterm.data.dataset import build_transforms
from midterm.data.vocab import load_vocab
from midterm.evaluate import load_model

app = Flask(__name__)

# Trạng thái model đang phục vụ. Lock để /load và /predict không giẫm chân nhau
# (Flask dev server bật threaded=True).
_state = {"ready": False, "checkpoint": None, "has_attention": False}
_lock = threading.Lock()
_device = pick_device()
_model = None
_cfg = None
_tokenizer = None
_idx_to_answer = None
_transform = None


def _checkpoint_dir() -> Path:
    return Path(Config().checkpoint_dir)


def available_checkpoints() -> list[str]:
    """Tên (stem) mọi file *.pt trong checkpoint_dir, sort để ổn định."""
    return sorted(p.stem for p in _checkpoint_dir().glob("*.pt"))


def resolve_default_checkpoint(preferred: str | None) -> str | None:
    """preferred nếu tồn tại; ngược lại cross_attention nếu có; ngược lại file
    đầu tiên; None nếu checkpoint_dir trống. (Hiện local mới có concat.pt nên
    fallback giúp app luôn lên được — heatmap chỉ tắt khi không phải cross_attn.)"""
    names = available_checkpoints()
    if preferred and preferred in names:
        return preferred
    if "cross_attention" in names:
        return "cross_attention"
    return names[0] if names else None


def load_checkpoint(name: str) -> None:
    """Load checkpoint theo tên stem vào state toàn cục. Raise nếu file không có."""
    global _model, _cfg, _tokenizer, _idx_to_answer, _transform
    path = _checkpoint_dir() / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    with _lock:
        _state["ready"] = False
        model, cfg = load_model(str(path), _device)
        vocab = load_vocab(cfg.vocab_path)
        _model = model
        _cfg = cfg
        _tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
        _idx_to_answer = {idx: ans for ans, idx in vocab.items()}
        _transform = build_transforms(cfg, train=False)
        _state["checkpoint"] = name
        _state["has_attention"] = cfg.fusion == "cross_attention"
        _state["ready"] = True


def render_heatmap(attn: torch.Tensor, image: Image.Image) -> str:
    """attn (1,49) → overlay jet trên ảnh 224×224 → PNG base64.

    Cùng cách dựng như demo.py: lưới 7×7 phóng to 32× bằng np.kron."""
    heat = attn.squeeze(0).reshape(7, 7).cpu().numpy()
    heat_big = np.kron(heat, np.ones((32, 32)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image.resize((224, 224)))
    ax.imshow(heat_big, cmap="jet", alpha=0.4)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_inference(image: Image.Image, question: str) -> dict:
    """Chạy model trên (ảnh, câu hỏi) → top-5 + heatmap (nếu cross_attention)."""
    pixel = _transform(image).unsqueeze(0).to(_device)
    tokens = _tokenizer(question, padding="max_length", truncation=True,
                        max_length=_cfg.max_question_len, return_tensors="pt")
    with torch.no_grad():
        logits, attn = _model(pixel,
                              tokens["input_ids"].to(_device),
                              tokens["attention_mask"].to(_device))
    probs = logits.softmax(dim=-1).squeeze(0)
    top = probs.topk(5)
    answers = [{"answer": _idx_to_answer[i], "prob": float(p)}
               for p, i in zip(top.values.tolist(), top.indices.tolist())]
    heatmap = render_heatmap(attn, image) if attn is not None else None
    return {"answers": answers, "heatmap": heatmap,
            "has_attention": attn is not None}


@app.get("/health")
def health():
    return jsonify(_state)


@app.get("/checkpoints")
def checkpoints():
    return jsonify({"checkpoints": available_checkpoints(),
                    "current": _state["checkpoint"]})


@app.post("/load")
def load():
    name = (request.get_json(silent=True) or {}).get("checkpoint")
    if not name:
        return jsonify({"error": "thiếu 'checkpoint'"}), 400
    try:
        load_checkpoint(name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify(_state)


@app.post("/predict")
def predict():
    if not _state["ready"]:
        return jsonify({"error": "model đang tải"}), 409
    if "image" not in request.files:
        return jsonify({"error": "thiếu file 'image'"}), 400
    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "thiếu 'question'"}), 400
    image = Image.open(request.files["image"].stream).convert("RGB")
    return jsonify(run_inference(image, question))


def main() -> None:
    parser = argparse.ArgumentParser(description="MedVQA sidecar server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default=None,
                        help="tên stem checkpoint ưu tiên (vd cross_attention)")
    args = parser.parse_args()

    default = resolve_default_checkpoint(args.checkpoint)
    if default is not None:
        # Load ở thread nền để Flask bind cổng ngay → Go poll /health được
        threading.Thread(target=load_checkpoint, args=(default,),
                         daemon=True).start()

    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Chạy test fast → pass**

Run:
```bash
source .venv/bin/activate && python -m pytest midterm/tests/test_serve.py -q -k "not integration"
```
Expected: 6 passed (các test fast; chưa có test integration).

- [ ] **Step 6: Thêm test integration (load model thật + /predict + /load)**

Thêm vào cuối `midterm/tests/test_serve.py`:
```python
# ---- Integration: load checkpoint thật (chậm ~30s, cần >=1 file .pt) ----

@pytest.fixture(scope="module")
def loaded_client():
    """Load 1 checkpoint thật vào state rồi trả test client. Module-scoped để
    chỉ load model một lần cho mọi test integration."""
    name = serve.resolve_default_checkpoint(None)
    assert name is not None, "cần ít nhất 1 checkpoint trong midterm/checkpoints/"
    serve.load_checkpoint(name)
    return serve.app.test_client()


@pytest.mark.integration
def test_health_ready_after_load(loaded_client):
    body = loaded_client.get("/health").get_json()
    assert body["ready"] is True
    assert body["checkpoint"] is not None


@pytest.mark.integration
def test_predict_returns_top5(loaded_client):
    img = Image.new("RGB", (256, 256), "gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = loaded_client.post("/predict", data={
        "image": (buf, "x.png"),
        "question": "is this normal?",
    }, content_type="multipart/form-data")
    body = resp.get_json()
    assert resp.status_code == 200
    assert len(body["answers"]) == 5
    assert all("answer" in a and "prob" in a for a in body["answers"])
    # prob giảm dần (topk)
    probs = [a["prob"] for a in body["answers"]]
    assert probs == sorted(probs, reverse=True)
    # heatmap khác null đúng khi checkpoint là cross_attention
    assert (body["heatmap"] is not None) == body["has_attention"]


@pytest.mark.integration
def test_load_invalid_checkpoint_404(loaded_client):
    resp = loaded_client.post("/load", json={"checkpoint": "khong_ton_tai"})
    assert resp.status_code == 404
```

Thêm `io` vào import đầu file test (đã có `base64`):
```python
import base64
import io
```

- [ ] **Step 7: Chạy test integration → pass**

Run:
```bash
source .venv/bin/activate && python -m pytest midterm/tests/test_serve.py -q -m integration
```
Expected: 3 passed (chậm ~30s vì load model thật). Nếu báo `unknown marker integration`: thêm file `pytest.ini` ở repo root với:
```ini
[pytest]
markers =
    integration: load model thật, chậm
```

- [ ] **Step 8: Verify thủ công server chạy thật (curl)**

Run (terminal 1):
```bash
source .venv/bin/activate && python -m midterm.serve --port 8765
```
Run (terminal 2, đợi ~30s cho model load):
```bash
curl -s localhost:8765/health
curl -s localhost:8765/checkpoints
```
Expected: `/health` `ready` chuyển `true`; `/checkpoints` liệt kê `concat`. Dừng server (Ctrl-C).

- [ ] **Step 9: Commit**

```bash
git add midterm/serve.py midterm/tests/test_serve.py requirements.txt pytest.ini
git commit -m "feat(midterm): Flask sidecar server for desktop demo inference"
```

---

## Task 2: Wails scaffold + Go backend (`app.go`, `main.go`) + Go tests

**Files:**
- Create (qua `wails init`): `midterm/desktop/` (gồm `main.go`, `app.go`, `wails.json`, `go.mod`, `frontend/`)
- Modify: `midterm/desktop/app.go` (thay toàn bộ)
- Modify: `midterm/desktop/main.go` (sửa Title/size + OnShutdown)
- Create: `midterm/desktop/app_test.go`

**Interfaces:**
- Consumes: HTTP contract của sidecar (Task 1).
- Produces (Wails bindings cho frontend, Task 3):
  - `Health() (HealthResp, error)` với `HealthResp{Ready bool; Checkpoint string; HasAttention bool}`
  - `Checkpoints() (CheckpointsResp, error)` với `CheckpointsResp{Checkpoints []string; Current string}`
  - `LoadCheckpoint(name string) (HealthResp, error)`
  - `Predict(imageB64 string, question string) (PredictResp, error)` với `PredictResp{Answers []Answer; Heatmap string; HasAttention bool}`, `Answer{Answer string; Prob float64}`

- [ ] **Step 1: Khởi tạo project Wails React-TS**

Run:
```bash
cd midterm && wails init -n desktop -t react-ts && cd ..
ls midterm/desktop
```
Expected: tạo `midterm/desktop/` có `main.go`, `app.go`, `wails.json`, `go.mod`, `frontend/`.

- [ ] **Step 2: Viết Go test cho bindings (httptest mock sidecar)**

Create `midterm/desktop/app_test.go`:
```go
package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// newTestApp trỏ baseURL vào một mock server thay cho sidecar thật.
func newTestApp(srv *httptest.Server) *App {
	a := NewApp()
	a.baseURL = srv.URL
	return a
}

func TestHealthParsesJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ready":true,"checkpoint":"cross_attention","has_attention":true}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	h, err := a.Health()
	if err != nil {
		t.Fatal(err)
	}
	if !h.Ready || h.Checkpoint != "cross_attention" || !h.HasAttention {
		t.Fatalf("parse sai: %+v", h)
	}
}

func TestHealthHandlesNullCheckpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ready":false,"checkpoint":null,"has_attention":false}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	h, err := a.Health()
	if err != nil {
		t.Fatal(err)
	}
	if h.Ready || h.Checkpoint != "" {
		t.Fatalf("null checkpoint phải thành \"\": %+v", h)
	}
}

func TestPredictSendsMultipartAndParses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("không parse được multipart: %v", err)
		}
		if r.FormValue("question") != "is there cardiomegaly?" {
			t.Errorf("question sai: %q", r.FormValue("question"))
		}
		if _, _, err := r.FormFile("image"); err != nil {
			t.Errorf("thiếu file image: %v", err)
		}
		w.Write([]byte(`{"answers":[{"answer":"no","prob":0.8},{"answer":"yes","prob":0.2}],"heatmap":"QUJD","has_attention":true}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	// "QUJD" là base64 của "ABC" → ảnh giả hợp lệ cho test
	pr, err := a.Predict("QUJD", "is there cardiomegaly?")
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Answers) != 2 || pr.Answers[0].Answer != "no" || !pr.HasAttention {
		t.Fatalf("parse sai: %+v", pr)
	}
}

func TestPredictErrorsOnNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(409)
		w.Write([]byte(`{"error":"model đang tải"}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	if _, err := a.Predict("QUJD", "q"); err == nil {
		t.Fatal("mong đợi lỗi khi status != 200")
	}
}
```

- [ ] **Step 3: Chạy test → fail (chưa có baseURL/bindings)**

Run:
```bash
cd midterm/desktop && go test ./... 2>&1 | head -20; cd ../..
```
Expected: FAIL biên dịch — `a.baseURL undefined`, `a.Health undefined`, v.v. (Nếu lỗi `pattern frontend/dist: no matching files` → chạy `mkdir -p midterm/desktop/frontend/dist && touch midterm/desktop/frontend/dist/index.html` rồi chạy lại.)

- [ ] **Step 4: Thay toàn bộ `midterm/desktop/app.go`**

Replace `midterm/desktop/app.go`:
```go
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// App quản lý vòng đời sidecar Python và là proxy mỏng giữa frontend và sidecar.
type App struct {
	ctx     context.Context
	baseURL string // http://127.0.0.1:<port>
	cmd     *exec.Cmd
	client  *http.Client
}

func NewApp() *App {
	return &App{client: &http.Client{Timeout: 120 * time.Second}}
}

type HealthResp struct {
	Ready        bool   `json:"ready"`
	Checkpoint   string `json:"checkpoint"`
	HasAttention bool   `json:"has_attention"`
}

type CheckpointsResp struct {
	Checkpoints []string `json:"checkpoints"`
	Current     string   `json:"current"`
}

type Answer struct {
	Answer string  `json:"answer"`
	Prob   float64 `json:"prob"`
}

type PredictResp struct {
	Answers      []Answer `json:"answers"`
	Heatmap      string   `json:"heatmap"`
	HasAttention bool     `json:"has_attention"`
}

func isDir(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}

// findRepoRoot đi ngược từ CWD lên trên tìm thư mục chứa cả ".venv" và "midterm".
// Override được bằng biến môi trường DEEPLEARNING_ROOT.
func findRepoRoot() (string, error) {
	if env := os.Getenv("DEEPLEARNING_ROOT"); env != "" {
		return env, nil
	}
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if isDir(filepath.Join(dir, ".venv")) && isDir(filepath.Join(dir, "midterm")) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("không tìm thấy repo root (.venv + midterm); đặt biến DEEPLEARNING_ROOT")
		}
		dir = parent
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	root, err := findRepoRoot()
	if err != nil {
		fmt.Println("startup error:", err)
		return
	}
	port, err := freePort()
	if err != nil {
		fmt.Println("startup error:", err)
		return
	}
	a.baseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	python := filepath.Join(root, ".venv", "bin", "python")
	a.cmd = exec.Command(python, "-m", "midterm.serve", "--port", fmt.Sprint(port),
		"--checkpoint", "cross_attention")
	a.cmd.Dir = root
	a.cmd.Stdout = os.Stdout
	a.cmd.Stderr = os.Stderr
	if err := a.cmd.Start(); err != nil {
		fmt.Println("không khởi động được sidecar:", err)
	}
}

func (a *App) shutdown(ctx context.Context) {
	if a.cmd != nil && a.cmd.Process != nil {
		_ = a.cmd.Process.Kill()
	}
}

func (a *App) getJSON(path string, out any) error {
	resp, err := a.client.Get(a.baseURL + path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(out)
}

func (a *App) Health() (HealthResp, error) {
	var h HealthResp
	err := a.getJSON("/health", &h)
	return h, err
}

func (a *App) Checkpoints() (CheckpointsResp, error) {
	var c CheckpointsResp
	err := a.getJSON("/checkpoints", &c)
	return c, err
}

func (a *App) LoadCheckpoint(name string) (HealthResp, error) {
	var h HealthResp
	body, _ := json.Marshal(map[string]string{"checkpoint": name})
	resp, err := a.client.Post(a.baseURL+"/load", "application/json", bytes.NewReader(body))
	if err != nil {
		return h, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return h, fmt.Errorf("load lỗi %d: %s", resp.StatusCode, string(b))
	}
	err = json.NewDecoder(resp.Body).Decode(&h)
	return h, err
}

func (a *App) Predict(imageB64 string, question string) (PredictResp, error) {
	var pr PredictResp
	raw, err := base64.StdEncoding.DecodeString(imageB64)
	if err != nil {
		return pr, fmt.Errorf("ảnh base64 không hợp lệ: %w", err)
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile("image", "upload.png")
	if err != nil {
		return pr, err
	}
	if _, err := fw.Write(raw); err != nil {
		return pr, err
	}
	_ = w.WriteField("question", question)
	w.Close()

	resp, err := a.client.Post(a.baseURL+"/predict", w.FormDataContentType(), &buf)
	if err != nil {
		return pr, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return pr, fmt.Errorf("predict lỗi %d: %s", resp.StatusCode, string(b))
	}
	err = json.NewDecoder(resp.Body).Decode(&pr)
	return pr, err
}
```

- [ ] **Step 5: Chạy Go test → pass**

Run:
```bash
cd midterm/desktop && go test ./... ; cd ../..
```
Expected: PASS (4 test trong `app_test.go`).

- [ ] **Step 6: Sửa `midterm/desktop/main.go` — Title, kích thước, OnShutdown**

Mở `midterm/desktop/main.go` (template tạo sẵn). Thực hiện đúng các sửa sau trong lời gọi `wails.Run(&options.App{...})`:
1. Đổi `Title:` thành `"MedVQA Demo — VQA-RAD"`.
2. Đặt `Width: 1100,` và `Height: 720,`.
3. Thêm dòng `OnShutdown: app.shutdown,` ngay dưới dòng `OnStartup: app.startup,`.

Phần còn lại (`//go:embed all:frontend/dist`, `AssetServer`, `Bind: []interface{}{app}`) giữ nguyên như template.

- [ ] **Step 7: Sinh bindings cho frontend**

Run:
```bash
cd midterm/desktop && wails generate module ; cd ../..
ls midterm/desktop/frontend/wailsjs/go/main
```
Expected: có `App.ts` (export `Health`, `Checkpoints`, `LoadCheckpoint`, `Predict`) và `frontend/wailsjs/go/models.ts` (namespace `main` với `HealthResp`, `CheckpointsResp`, `PredictResp`, `Answer`).

- [ ] **Step 8: Build Go để chắc chắn biên dịch**

Run:
```bash
cd midterm/desktop && go build ./... ; cd ../..
```
Expected: build thành công, không lỗi. (Nếu lỗi embed `frontend/dist`: `cd midterm/desktop/frontend && npm install && npm run build`, rồi build lại.)

- [ ] **Step 9: Commit**

```bash
git add midterm/desktop
git commit -m "feat(midterm): Wails Go backend — sidecar lifecycle + HTTP proxy bindings"
```

---

## Task 3: React frontend (apply frontend-design skill)

**Files:**
- Create: `midterm/desktop/frontend/src/lib.ts`
- Create: `midterm/desktop/frontend/src/lib.test.ts`
- Modify: `midterm/desktop/frontend/src/App.tsx` (thay toàn bộ)
- Modify: `midterm/desktop/frontend/src/App.css` (style)
- Modify: `midterm/desktop/frontend/package.json` (thêm devDep vitest)

**Interfaces:**
- Consumes: bindings `Health`, `Checkpoints`, `LoadCheckpoint`, `Predict` từ `../wailsjs/go/main/App`; types từ `../wailsjs/go/models`.
- Produces: helper thuần `stripDataUrl(dataUrl: string) -> string`, `formatProb(p: number) -> string`.

- [ ] **Step 1: Cài vitest**

Run:
```bash
cd midterm/desktop/frontend && npm install && npm install -D vitest && cd ../../..
```
Expected: cài xong. Thêm script vào `frontend/package.json` mục `"scripts"`: `"test": "vitest run"`.

- [ ] **Step 2: Viết test cho helper thuần**

Create `midterm/desktop/frontend/src/lib.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { formatProb, stripDataUrl } from "./lib";

describe("stripDataUrl", () => {
  it("bỏ tiền tố data URL, giữ base64", () => {
    expect(stripDataUrl("data:image/png;base64,QUJD")).toBe("QUJD");
  });
  it("trả nguyên chuỗi nếu không có tiền tố", () => {
    expect(stripDataUrl("QUJD")).toBe("QUJD");
  });
});

describe("formatProb", () => {
  it("format xác suất thành phần trăm 1 chữ số", () => {
    expect(formatProb(0.8123)).toBe("81.2%");
  });
});
```

- [ ] **Step 3: Chạy test → fail (chưa có lib.ts)**

Run:
```bash
cd midterm/desktop/frontend && npm test 2>&1 | head -15; cd ../../..
```
Expected: FAIL — không resolve được `./lib`.

- [ ] **Step 4: Viết `lib.ts`**

Create `midterm/desktop/frontend/src/lib.ts`:
```ts
/** Bỏ tiền tố "data:...;base64," của FileReader, chỉ giữ phần base64 thuần
 *  để gửi sang Go (Predict nhận base64 không tiền tố). */
export function stripDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

/** 0.8123 → "81.2%" */
export function formatProb(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}
```

- [ ] **Step 5: Chạy test → pass**

Run:
```bash
cd midterm/desktop/frontend && npm test ; cd ../../..
```
Expected: 3 passed.

- [ ] **Step 6: Áp dụng skill frontend-design cho phần nhìn**

Invoke skill `frontend-design` để chốt hệ token (màu, typography, spacing) và phong cách cho app demo y khoa (gọn, chuyên nghiệp, dễ đọc trước giám khảo). Output của skill dùng để viết `App.css` ở Step 8. Ghi lại bảng màu/typography đã chọn vào đầu `App.css` dưới dạng comment.

- [ ] **Step 7: Thay toàn bộ `App.tsx` (state machine + UI)**

Replace `midterm/desktop/frontend/src/App.tsx`:
```tsx
import { useEffect, useRef, useState } from "react";
import {
  Checkpoints,
  Health,
  LoadCheckpoint,
  Predict,
} from "../wailsjs/go/main/App";
import { main } from "../wailsjs/go/models";
import { formatProb, stripDataUrl } from "./lib";
import "./App.css";

type Phase = "loading" | "ready" | "predicting" | "switching";

function App() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [current, setCurrent] = useState("");
  const [hasAttention, setHasAttention] = useState(false);
  const [imageB64, setImageB64] = useState("");
  const [preview, setPreview] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<main.PredictResp | null>(null);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  // Poll /health khi khởi động cho tới khi model sẵn sàng
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const h = await Health();
        if (cancelled) return;
        if (h.ready) {
          setHasAttention(h.has_attention);
          const c = await Checkpoints();
          setCheckpoints(c.checkpoints);
          setCurrent(c.current);
          setPhase("ready");
          return;
        }
      } catch {
        /* sidecar chưa lên — thử lại */
      }
      if (!cancelled) setTimeout(poll, 1000);
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  const onFile = (f: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setPreview(dataUrl);
      setImageB64(stripDataUrl(dataUrl));
      setResult(null);
    };
    reader.readAsDataURL(f);
  };

  const onRun = async () => {
    if (!imageB64 || !question.trim()) return;
    setPhase("predicting");
    setError("");
    try {
      setResult(await Predict(imageB64, question));
    } catch (e) {
      setError(String(e));
    } finally {
      setPhase("ready");
    }
  };

  const onSwitch = async (name: string) => {
    if (name === current) return;
    setPhase("switching");
    setResult(null);
    setError("");
    try {
      const h = await LoadCheckpoint(name);
      setCurrent(h.checkpoint);
      setHasAttention(h.has_attention);
    } catch (e) {
      setError(String(e));
    } finally {
      setPhase("ready");
    }
  };

  const busy = phase === "loading" || phase === "predicting" || phase === "switching";
  const canRun = phase === "ready" && !!imageB64 && !!question.trim();
  const top1 = result?.answers?.[0];

  return (
    <div className="app">
      <header className="topbar">
        <h1>MedVQA Demo — VQA-RAD</h1>
        <div className="status">
          <label>
            Checkpoint:&nbsp;
            <select
              value={current}
              disabled={busy}
              onChange={(e) => onSwitch(e.target.value)}
            >
              {checkpoints.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>
          <span className={`dot ${phase === "ready" ? "ok" : "wait"}`} />
          <span>
            {phase === "loading" && "Đang tải model…"}
            {phase === "switching" && "Đang đổi checkpoint…"}
            {phase === "predicting" && "Đang suy luận…"}
            {phase === "ready" && "Sẵn sàng"}
          </span>
        </div>
      </header>

      <main className="grid">
        <section className="left">
          <div
            className="dropzone"
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) onFile(f);
            }}
          >
            {preview ? (
              <img src={preview} alt="preview" />
            ) : (
              <span>Kéo-thả hoặc bấm để chọn ảnh</span>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onFile(f);
              }}
            />
          </div>

          <label className="field">
            Câu hỏi (tiếng Anh):
            <input
              type="text"
              value={question}
              placeholder="is there cardiomegaly?"
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onRun()}
            />
          </label>

          <button className="run" disabled={!canRun} onClick={onRun}>
            {phase === "predicting" ? "Đang chạy…" : "Run"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="right">
          {top1 && (
            <div className="answer">
              <span className="label">Đáp án</span>
              <span className="value">{top1.answer}</span>
            </div>
          )}
          {result && (
            <div className="top5">
              <span className="label">Top-5</span>
              {result.answers.map((a) => (
                <div className="bar-row" key={a.answer}>
                  <span className="name">{a.answer}</span>
                  <div className="bar">
                    <div className="fill" style={{ width: `${a.prob * 100}%` }} />
                  </div>
                  <span className="pct">{formatProb(a.prob)}</span>
                </div>
              ))}
            </div>
          )}
          {result &&
            (result.has_attention && result.heatmap ? (
              <div className="heatmap">
                <span className="label">Attention heatmap</span>
                <img src={`data:image/png;base64,${result.heatmap}`} alt="heatmap" />
              </div>
            ) : (
              <p className="note">
                Checkpoint “{current}” không có attention để vẽ heatmap.
              </p>
            ))}
        </section>
      </main>
    </div>
  );
}

export default App;
```

- [ ] **Step 8: Viết `App.css` theo token của frontend-design**

Replace `midterm/desktop/frontend/src/App.css` bằng style dựa trên hệ token đã chốt ở Step 6. Yêu cầu bố cục: `.app` cao toàn màn hình, `.topbar` flex giữa tiêu đề và `.status`; `.grid` chia 2 cột (trái/phải); `.dropzone` viền nét đứt, ảnh preview `max-width:100%`; `.bar .fill` nền màu nhấn theo prob; `.answer .value` cỡ chữ lớn (~2rem). (Nội dung CSS cụ thể do skill frontend-design quyết định; mọi class trên phải được style.)

- [ ] **Step 9: Verify build frontend + test**

Run:
```bash
cd midterm/desktop/frontend && npm test && npm run build ; cd ../../..
```
Expected: vitest pass; `npm run build` tạo `frontend/dist` không lỗi TypeScript.

- [ ] **Step 10: Commit**

```bash
git add midterm/desktop/frontend
git commit -m "feat(midterm): React UI for desktop demo — upload, predict, heatmap, checkpoint switch"
```

---

## Task 4: End-to-end manual verification + README

**Files:**
- Create: `midterm/desktop/README.md`

- [ ] **Step 1: Chạy app end-to-end bằng `wails dev`**

Run:
```bash
cd midterm/desktop && wails dev
```
Expected: cửa sổ app mở; thanh trạng thái hiện "Đang tải model…" rồi chuyển "Sẵn sàng" sau ~15–30s; dropdown checkpoint có `concat` (và `cross_attention` nếu đã train).

- [ ] **Step 2: Verify luồng hỏi-đáp**

Trong app: chọn một ảnh test (vd `midterm/data/...` hoặc bất kỳ ảnh X-quang), gõ `is there cardiomegaly?`, bấm **Run**.
Expected: hiện đáp án top-1 (chữ to) + 5 thanh xác suất giảm dần. Nếu checkpoint là `cross_attention` → có ảnh heatmap; nếu `concat` → hiện ghi chú "không có attention".

- [ ] **Step 3: Verify đổi checkpoint**

Đổi dropdown sang checkpoint khác.
Expected: trạng thái chuyển "Đang đổi checkpoint…" → "Sẵn sàng"; kết quả cũ bị xóa; panel heatmap ẩn/hiện đúng theo `has_attention` của checkpoint mới. Đóng cửa sổ → tiến trình `python -m midterm.serve` cũng bị kill (kiểm tra `pgrep -f midterm.serve` trống).

- [ ] **Step 4: Viết `midterm/desktop/README.md`**

Create `midterm/desktop/README.md`:
```markdown
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
```

- [ ] **Step 5: Commit**

```bash
git add midterm/desktop/README.md
git commit -m "docs(midterm): desktop demo app README + manual verification"
```

---

## Self-Review

**Spec coverage:**
- Sidecar Flask + 4 endpoint (spec §4) → Task 1. ✅
- Heatmap đúng logic demo.py (spec §2, §4.2) → Task 1 `render_heatmap`. ✅
- Checkpoint mặc định robust/fallback (spec §4.1) → Task 1 `resolve_default_checkpoint` + test. ✅
- Go spawn sidecar, free port, repo-root, kill khi shutdown (spec §5.1) → Task 2. ✅
- Bindings Health/Checkpoints/LoadCheckpoint/Predict (spec §5.2) → Task 2. ✅
- React UI 2 cột, dropdown checkpoint, top-1+top-5, heatmap có điều kiện, trạng thái loading (spec §6, §7) → Task 3. ✅
- frontend-design cho UI đẹp (yêu cầu người dùng) → Task 3 Step 6. ✅
- Xử lý lỗi: thiếu ảnh/câu hỏi (nút khóa), predict lỗi, đang load (409), checkpoint không attention (spec §8) → Task 2 (409, error) + Task 3 (canRun, note). ✅
- Kiểm thử: sidecar pytest, Go binding mock, thủ công e2e (spec §9) → Task 1/2/4. ✅

**Placeholder scan:** Không có TBD/TODO; mọi step có code/command cụ thể. CSS cụ thể giao cho frontend-design (Task 3 Step 6/8) — đây là quyết định thiết kế có chủ đích, không phải placeholder (class cần style đã liệt kê rõ).

**Type consistency:** `HealthResp{Ready,Checkpoint,HasAttention}`, `CheckpointsResp{Checkpoints,Current}`, `PredictResp{Answers,Heatmap,HasAttention}`, `Answer{Answer,Prob}` nhất quán giữa Go (Task 2), JSON sidecar (Task 1), và frontend `main.PredictResp` (Task 3). Hàm Python `available_checkpoints`/`resolve_default_checkpoint`/`render_heatmap`/`load_checkpoint` dùng đúng tên giữa serve.py và test. Helper `stripDataUrl`/`formatProb` khớp giữa lib.ts, lib.test.ts, App.tsx. ✅
