"""Test sidecar. Phần fast không cần load model; phần integration (cuối file)
load checkpoint thật nên chậm (~30s) và cần checkpoints/ có ít nhất 1 file .pt."""
import base64
import io

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


def test_load_missing_checkpoint_field_400():
    client = serve.app.test_client()
    resp = client.post("/load", json={})
    assert resp.status_code == 400


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
