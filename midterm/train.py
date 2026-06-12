"""Training loop cho MedVQA.

Ba thí nghiệm chính của báo cáo (chỉ khác fusion, cùng seed):

    python -m midterm.train --fusion concat
    python -m midterm.train --fusion hadamard
    python -m midterm.train --fusion cross_attention

Smoke test local (subset 128 mẫu, 2 epoch, run_name có hậu tố _smoke để không
ghi đè checkpoint thật):

    python -m midterm.train --fusion concat --smoke

Mỗi run lưu: outputs/<run_name>/{config.json, history.json, curves.png}
và checkpoint tốt nhất (theo val accuracy) tại checkpoints/<run_name>.pt.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render PNG không cần GUI (chạy được trên server/Colab)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from midterm.config import Config, pick_device
from midterm.data.dataset import VQARadDataset, load_splits
from midterm.data.vocab import load_vocab
from midterm.models.vqa_model import VQAModel


def set_seed(seed: int) -> None:
    """Fix mọi nguồn ngẫu nhiên: 3 thí nghiệm fusion chỉ khác nhau ở fusion,
    không khác ở khởi tạo hay thứ tự shuffle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_validation(model, loader, device):
    """Trả về (accuracy, loss trung bình) trên loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(images, input_ids, attention_mask)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def plot_curves(history: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch"), ax1.set_title("Loss"), ax1.legend()
    ax2.plot(epochs, history["val_acc"])
    ax2.set_xlabel("epoch"), ax2.set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: Config, smoke: bool = False) -> float:
    set_seed(cfg.seed)
    device = pick_device()
    print(f"Device: {device} | run: {cfg.run_name} | fusion: {cfg.fusion}"
          f" | text_pool: {cfg.text_pool}")

    vocab = load_vocab(cfg.vocab_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    train_hf, val_hf, _ = load_splits(cfg)
    train_ds = VQARadDataset(train_hf, tokenizer, vocab, cfg, train=True)
    val_ds = VQARadDataset(val_hf, tokenizer, vocab, cfg, train=False)
    if smoke:  # subset nhỏ: chỉ kiểm tra pipeline chạy end-to-end
        train_ds = Subset(train_ds, range(128))
        val_ds = Subset(val_ds, range(32))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = VQAModel(cfg, num_classes=len(vocab)).to(device)
    total, trainable = model.count_parameters()
    print(f"Tham số: {total / 1e6:.0f}M tổng | {trainable / 1e6:.1f}M trainable")

    # Hai param group: phần tự xây (fusion + projection + head) học LR cao;
    # layer4 của ResNet (nếu unfreeze) học LR thấp hơn 100 lần để không phá
    # feature pretrained bằng gradient lớn lúc đầu.
    backbone_params = [p for p in model.image_encoder.backbone.parameters()
                       if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    new_params = [p for p in model.trainable_parameters()
                  if id(p) not in backbone_ids]
    param_groups = [{"params": new_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, epochs_no_improve = 0.0, 0
    max_epochs = 2 if smoke else cfg.max_epochs

    for epoch in range(1, max_epochs + 1):
        model.train()  # encoder tự ghim eval bên trong (xem image/text encoder)
        loss_sum, seen = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        val_acc, val_loss = run_validation(model, val_loader, device)
        history["train_loss"].append(loss_sum / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d} | train loss {loss_sum / seen:.4f}"
              f" | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

        # Early stopping theo val overall accuracy; lưu checkpoint tốt nhất
        if val_acc > best_acc:
            best_acc, epochs_no_improve = val_acc, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
                "num_classes": len(vocab),
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping tại epoch {epoch} (patience {cfg.patience})")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, out_dir / "curves.png")
    print(f"Best val acc: {best_acc:.4f} | checkpoint: {ckpt_path}")
    return best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MedVQA trên VQA-RAD")
    parser.add_argument("--fusion", default="concat",
                        choices=["concat", "hadamard", "cross_attention"])
    parser.add_argument("--run-name", default="")
    parser.add_argument("--text-pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="subset 128 mẫu + 2 epoch: kiểm tra pipeline end-to-end")
    args = parser.parse_args()

    run_name = args.run_name or (f"{args.fusion}_smoke" if args.smoke else args.fusion)
    cfg = Config(fusion=args.fusion, run_name=run_name, text_pool=args.text_pool,
                 unfreeze_last_block=args.unfreeze_last_block)
    if args.smoke:
        cfg.batch_size = 16
        cfg.num_workers = 0
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
