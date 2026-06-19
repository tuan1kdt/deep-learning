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
