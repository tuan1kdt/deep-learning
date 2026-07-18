"""Fine-tune end-to-end (mở rộng): gỡ hai hạn chế đã khai trong báo cáo.

Toàn bộ pipeline chính train decoder trên feature tính trước — đổi lấy tốc
độ, chấp nhận (1) encoder đóng băng, (2) không augmentation. Bước này gỡ cả
hai: nạp lại checkpoint decoder tốt nhất, gắn ResNet thật vào trước, mở băng
RIÊNG layer4 (BatchNorm toàn mạng ghim eval — thống kê ImageNet giữ nguyên,
chỉ trọng số conv của layer4 học tiếp), train trên ảnh thô với
RandomResizedCrop + flip, mixed precision để tận dụng GPU.

Vì feature giờ phụ thuộc backbone đã tinh chỉnh, evaluate.py (đọc feature
tính trước) không dùng được — script tự trích feature test bằng backbone
mới rồi chấm cùng bộ metric.

Chạy: .venv/bin/python -m final.finetune --checkpoint final/checkpoints/lstm_r101.pt
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from final.config import Config, pick_device
from final.data.download import load_captioning_dataset
from final.data.features import _MEAN, _STD, build_backbone, pool_to_regions
from final.data.vocab import BOS_ID, EOS_ID, PAD_ID, Vocab
from final.evaluate import compute_metrics, generate, load_model_from_checkpoint
from final.train import make_criterion, set_seed

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


class RawCaptionDataset(Dataset):
    """Như CaptionDataset nhưng trả ảnh thô (transform mỗi lần lấy mẫu —
    augmentation khác nhau giữa các epoch, điều feature tính trước không làm được)."""

    def __init__(self, rows, captions, vocab: Vocab, max_words: int, train: bool):
        assert len(rows) == len(captions)
        self.rows = rows
        self.tf = TRAIN_TF if train else EVAL_TF
        self.max_words = max_words
        self.samples = [(i, vocab.encode(c, max_words))
                        for i, group in enumerate(captions) for c in group]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_idx, ids = self.samples[i]
        pad = [PAD_ID] * (self.max_words - len(ids))
        return {
            "image": self.tf(self.rows[img_idx]["image"].convert("RGB")),
            "cap_in": torch.tensor([BOS_ID] + ids + pad, dtype=torch.long),
            "cap_out": torch.tensor(ids + [EOS_ID] + pad, dtype=torch.long),
        }


def unfreeze_layer4(backbone: nn.Sequential) -> list[nn.Parameter]:
    """Mở băng conv của layer4 (khối cuối); BatchNorm vẫn eval + đóng băng
    affine để thống kê chuẩn hóa không trôi trên dataset nhỏ."""
    params = []
    for m in backbone[-1].modules():
        if isinstance(m, nn.Conv2d):
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
    return params


def encode_batch(backbone, images, device, amp: bool):
    with torch.autocast("cuda", enabled=amp):
        return pool_to_regions(backbone(images.to(device))).float()


def extract_test_features(backbone, rows, device, amp: bool) -> torch.Tensor:
    tf = EVAL_TF
    feats, batch = [], []
    with torch.no_grad():
        for i in range(len(rows)):
            batch.append(tf(rows[i]["image"].convert("RGB")))
            if len(batch) == 64 or i == len(rows) - 1:
                x = torch.stack(batch)
                feats.append(encode_batch(backbone, x, device, amp).cpu())
                batch = []
    return torch.cat(feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--smoke", action="store_true", help="512 mẫu, 1 epoch")
    args = parser.parse_args()

    device = pick_device()
    amp = device.type == "cuda"
    model, cfg, vocab = load_model_from_checkpoint(args.checkpoint, device)
    set_seed(cfg.seed)
    run_name = f"{cfg.run_name}_ft"
    print(f"Device: {device} | run: {run_name} | encoder: {cfg.encoder}"
          f" (mở băng layer4) | amp: {amp}")

    ds = load_captioning_dataset(cfg)
    caps = {s: json.loads(cfg.captions_path(s).read_text())
            for s in ("train", "validation", "test")}
    train_ds = RawCaptionDataset(ds["train"], caps["train"], vocab,
                                 cfg.max_words, train=True)
    val_ds = RawCaptionDataset(ds["validation"], caps["validation"], vocab,
                               cfg.max_words, train=False)
    if args.smoke:
        train_ds, val_ds = Subset(train_ds, range(512)), Subset(val_ds, range(128))
        args.epochs = 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=amp)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=amp)

    backbone = build_backbone(cfg.encoder).to(device)  # eval + frozen sẵn
    enc_params = unfreeze_layer4(backbone)
    model.train()
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr},
         {"params": model.parameters(), "lr": args.lr}],
        weight_decay=1e-2)
    criterion = make_criterion(cfg.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=amp)
    n_enc = sum(p.numel() for p in enc_params)
    print(f"Tham số mở băng encoder: {n_enc / 1e6:.1f}M"
          f" | decoder: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_tokens = 0.0, 0
        for batch in train_loader:
            cap_in = batch["cap_in"].to(device)
            cap_out = batch["cap_out"].to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=amp):
                feats = pool_to_regions(backbone(batch["image"].to(device)))
                logits, _ = model(feats.float(), cap_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 cap_out.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                enc_params + list(model.parameters()), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n = (cap_out != PAD_ID).sum().item()
            loss_sum += loss.item() * n
            n_tokens += n

        model.eval()
        vl_sum, vl_tok = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                cap_out = batch["cap_out"].to(device)
                feats = encode_batch(backbone, batch["image"], device, amp)
                logits, _ = model(feats, batch["cap_in"].to(device))
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 cap_out.reshape(-1))
                n = (cap_out != PAD_ID).sum().item()
                vl_sum += loss.item() * n
                vl_tok += n
        val_loss = vl_sum / vl_tok
        history["train_loss"].append(loss_sum / n_tokens)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d} | train {loss_sum / n_tokens:.4f}"
              f" | val {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "backbone_state": backbone.state_dict(),
                "config": {**cfg.to_dict(), "run_name": run_name},
                "vocab_size": len(vocab),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)

    # Nạp lại trạng thái tốt nhất rồi tự đánh giá (feature phụ thuộc backbone
    # đã tinh chỉnh nên không dùng evaluate.py với feature tính trước được)
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    backbone.load_state_dict(best["backbone_state"])
    model.eval()
    test_rows = ds["test"]
    if args.smoke:
        test_rows = test_rows.select(range(64))
    feats = extract_test_features(backbone, test_rows, device, amp)
    refs = caps["test"][:len(test_rows)]
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    rows_md = ["| mode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |",
               "|---|---|---|---|---|---|---|"]
    for mode in ("greedy", "beam3", "beam5"):
        hyps = generate(model, feats, mode, cfg, vocab, device)
        (out_dir / f"hyps_{mode}.json").write_text(
            json.dumps(hyps, ensure_ascii=False, indent=2))
        m = compute_metrics(hyps, refs)
        rows_md.append(f"| {mode} | {m['bleu1']:.3f} | {m['bleu2']:.3f} "
                       f"| {m['bleu3']:.3f} | {m['bleu4']:.3f} "
                       f"| {m['rouge_l']:.3f} | {m['cider']:.3f} |")
        print(rows_md[-1])
    table = "\n".join(rows_md)
    (out_dir / "eval.md").write_text(f"# Eval {run_name}\n\n{table}\n")
    print(f"\n{table}\n→ {out_dir / 'eval.md'}")


if __name__ == "__main__":
    main()
