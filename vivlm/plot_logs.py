"""Vẽ loss curve + biểu đồ util/VRAM — bằng chứng 'tận dụng hết trainbox'."""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(csv_path):
    with open(csv_path) as f:
        return list(csv.reader(f))


def plot_pretrain(csv_path, out_png):
    rows = _read(csv_path)[1:]
    tr = [(int(r[0]), float(r[2])) for r in rows if r[1] == "train"]
    va = [(int(r[0]), float(r[2])) for r in rows if r[1] == "val"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(*zip(*tr), label="train", alpha=0.7)
    if va:
        ax.plot(*zip(*va), "o-", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Pretrain ViGPT-nano (3B token tiếng Việt)")
    ax.legend()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def plot_sft(csv_path, out_png):
    rows = _read(csv_path)[1:]
    fig, ax = plt.subplots(figsize=(8, 5))
    for phase in ("projector", "full"):
        pts = [(int(r[1]), float(r[2])) for r in rows if r[0] == phase]
        if pts:
            ax.plot(*zip(*pts), label=f"phase {phase}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("SFT hai pha")
    ax.legend()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def plot_gpu(csv_path, out_png):
    rows = [r for r in _read(csv_path) if len(r) >= 3]
    util = [float(r[1]) for r in rows]
    mem = [float(r[2]) / 1024 for r in rows]         # MiB -> GiB
    t = [i * 0.5 / 60 for i in range(len(rows))]     # phút -> giờ (mẫu 30s)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, util, color="tab:blue", alpha=0.8, label="GPU util %")
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("giờ")
    ax1.set_ylabel("GPU util (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(t, mem, color="tab:red", alpha=0.8, label="VRAM GiB")
    ax2.set_ylim(0, 24)
    ax2.set_ylabel("VRAM (GiB)", color="tab:red")
    ax1.set_title("Trainbox RTX PRO 4000 24GB — util & VRAM")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["pretrain", "sft", "gpu"],
                    required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    {"pretrain": plot_pretrain, "sft": plot_sft,
     "gpu": plot_gpu}[args.which](args.csv, args.out)


if __name__ == "__main__":
    main()
