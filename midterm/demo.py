"""Demo inference: một ảnh + một câu hỏi → top-5 đáp án kèm xác suất.

Chạy: python -m midterm.demo --checkpoint midterm/checkpoints/cross_attention.pt \
          --image chest.jpg --question "is there cardiomegaly?"

Với checkpoint cross_attention: lưu thêm attention overlay (heatmap 7×7 phóng
to chồng lên ảnh) — minh họa model "nhìn" vào vùng nào khi trả lời, dùng cho
báo cáo và vấn đáp.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from midterm.config import pick_device
from midterm.data.dataset import build_transforms
from midterm.data.vocab import load_vocab
from midterm.evaluate import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="MedVQA demo inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", default="demo_attention.png",
                        help="đường dẫn lưu attention overlay (chỉ cross_attention)")
    args = parser.parse_args()

    device = pick_device()
    model, cfg = load_model(args.checkpoint, device)
    vocab = load_vocab(cfg.vocab_path)
    idx_to_answer = {idx: ans for ans, idx in vocab.items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    image = Image.open(args.image).convert("RGB")
    pixel = build_transforms(cfg, train=False)(image).unsqueeze(0).to(device)
    tokens = tokenizer(args.question, padding="max_length", truncation=True,
                       max_length=cfg.max_question_len, return_tensors="pt")

    with torch.no_grad():
        logits, attn = model(pixel,
                             tokens["input_ids"].to(device),
                             tokens["attention_mask"].to(device))

    probs = logits.softmax(dim=-1).squeeze(0)
    top = probs.topk(5)
    print(f"Q: {args.question}")
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {idx_to_answer[idx]:<30s} {prob:.3f}")

    if attn is not None:
        # attn (1, 49) → lưới 7×7 → phóng to 32× bằng np.kron → 224×224
        heat = attn.squeeze(0).reshape(7, 7).cpu().numpy()
        heat_big = np.kron(heat, np.ones((32, 32)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image.resize((224, 224)))
        ax.imshow(heat_big, cmap="jet", alpha=0.4)
        ax.axis("off")
        ax.set_title(args.question, fontsize=9)
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Attention overlay → {args.out}")


if __name__ == "__main__":
    main()
