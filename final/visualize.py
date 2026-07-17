"""Ba loại hình cho báo cáo:

1. attention — heatmap "từ nào nhìn vùng nào": greedy decode một ảnh test,
   attention (T,49) → 7×7 → phóng to đè lên ảnh gốc. Hình đinh của báo cáo.
2. curves — loss train/val của nhiều run trên cùng một trục để so sánh
   LSTM vs Transformer.
3. samples — bảng markdown caption của 3 ảnh val cố định qua các epoch
   (lấy từ history.json) — "model học như thế nào" nhìn thấy được.

Chạy:
  .venv/bin/python -m final.visualize --what attention --checkpoint final/checkpoints/lstm.pt --indices 0,5,17
  .venv/bin/python -m final.visualize --what curves --runs lstm,transformer
  .venv/bin/python -m final.visualize --what samples --runs lstm
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from final.config import Config, pick_device
from final.data.dataset import load_eval_data


def attention_figure(image, words: list[str], attn: torch.Tensor):
    """image: PIL 224×224 (hoặc bất kỳ, sẽ resize); words: T từ đã sinh;
    attn: (T,49). Trả Figure: ảnh gốc + mỗi từ một ô heatmap."""
    image = image.convert("RGB").resize((224, 224))
    n = len(words) + 1
    cols = min(n, 5)
    fig_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(fig_rows, cols, figsize=(2.4 * cols, 2.6 * fig_rows))
    axes = np.atleast_1d(axes).ravel()
    axes[0].imshow(image)
    axes[0].set_title("ảnh gốc", fontsize=9)
    for t, word in enumerate(words):
        ax = axes[t + 1]
        grid = attn[t].reshape(7, 7).detach().cpu().numpy()
        # upsample 7×7 → 224×224 bằng lặp khối (đủ tốt cho minh họa)
        heat = np.kron(grid, np.ones((32, 32)))
        ax.imshow(image)
        ax.imshow(heat, alpha=0.5, cmap="jet")
        ax.set_title(word, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def cmd_attention(args) -> None:
    from datasets import load_from_disk

    from final.evaluate import load_model_from_checkpoint
    from final.models.decoding import greedy_decode

    device = pick_device()
    model, cfg, vocab = load_model_from_checkpoint(args.checkpoint, device)
    feats, _ = load_eval_data(cfg, "test")
    ds_test = load_from_disk(cfg.dataset_dir)["test"]
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in [int(i) for i in args.indices.split(",")]:
        seqs, attn = greedy_decode(model, feats[idx:idx + 1].to(device),
                                   cfg.max_len)
        words = [vocab.id2word[i] for i in seqs[0]]
        # attn trả về theo bước decode; bước t sinh ra từ t → lấy T bước đầu
        fig = attention_figure(ds_test[idx]["image"], words,
                               attn[0, :len(words)].cpu())
        path = out_dir / f"attention_{idx}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[{idx}] \"{' '.join(words)}\" → {path}")


def cmd_curves(args) -> None:
    cfg = Config()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in args.runs.split(","):
        hist = json.loads((Path(cfg.output_dir) / run / "history.json").read_text())
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], "--", label=f"{run} train")
        ax.plot(epochs, hist["val_loss"], "-", label=f"{run} val")
    ax.set_xlabel("epoch"), ax.set_ylabel("cross-entropy/token"), ax.legend()
    ax.set_title("So sánh learning curves")
    fig.tight_layout()
    out = Path(cfg.output_dir) / f"curves_{args.runs.replace(',', '_vs_')}.png"
    fig.savefig(out, dpi=120)
    print(f"→ {out}")


def cmd_samples(args) -> None:
    cfg = Config()
    for run in args.runs.split(","):
        hist = json.loads((Path(cfg.output_dir) / run / "history.json").read_text())
        lines = [f"# Caption 3 ảnh val cố định theo epoch — {run}", "",
                 "| epoch | ảnh 0 | ảnh 1 | ảnh 2 |", "|---|---|---|---|"]
        for ep, sams in enumerate(hist["samples"], start=1):
            lines.append(f"| {ep} | " + " | ".join(sams) + " |")
        out = Path(cfg.output_dir) / run / "samples_by_epoch.md"
        out.write_text("\n".join(lines) + "\n")
        print(f"→ {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", required=True,
                        choices=["attention", "curves", "samples"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--indices", default="0,1,2")
    parser.add_argument("--runs", default="lstm,transformer")
    args = parser.parse_args()
    {"attention": cmd_attention, "curves": cmd_curves,
     "samples": cmd_samples}[args.what](args)


if __name__ == "__main__":
    main()
