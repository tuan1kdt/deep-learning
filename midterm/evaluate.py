"""Đánh giá checkpoint trên test split (451 mẫu) — chỉ chạy ở bước cuối.

Ba số liệu chuẩn của VQA-RAD:
- overall: accuracy trên toàn bộ test
- closed:  accuracy trên câu hỏi yes/no (đáp án chuẩn hóa thuộc {yes, no})
- open:    accuracy trên phần còn lại

Chạy: python -m midterm.evaluate --checkpoint midterm/checkpoints/concat.pt
Kết quả chi tiết lưu vào outputs/<run_name>/test_results.json — nguyên liệu
cho phần phân tích lỗi của báo cáo.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from midterm.config import Config, pick_device
from midterm.data.dataset import VQARadDataset, load_splits
from midterm.data.vocab import load_vocab
from midterm.models.vqa_model import VQAModel


def load_model(checkpoint_path: str, device):
    """Dựng lại model từ config lưu trong checkpoint — evaluate/demo không cần
    biết run đó dùng fusion hay text_pool nào."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = VQAModel(cfg, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def evaluate(checkpoint: str) -> dict:
    device = pick_device()
    model, cfg = load_model(checkpoint, device)
    vocab = load_vocab(cfg.vocab_path)
    idx_to_answer = {idx: ans for ans, idx in vocab.items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    _, _, test_hf = load_splits(cfg)
    test_ds = VQARadDataset(test_hf, tokenizer, vocab, cfg, train=False)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0)

    rows = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["image"].to(device),
                              batch["input_ids"].to(device),
                              batch["attention_mask"].to(device))
            preds = logits.argmax(dim=-1).cpu()
            for question, answer, label, pred in zip(
                    batch["question"], batch["answer"], batch["label"], preds):
                rows.append({
                    "question": question,
                    "answer": answer,
                    "pred": idx_to_answer[pred.item()],
                    # label -1 (đáp án ngoài vocab) không bao giờ khớp → tính sai
                    "correct": pred.item() == label.item(),
                    "closed": answer in ("yes", "no"),
                })

    def accuracy(subset):
        return sum(r["correct"] for r in subset) / len(subset) if subset else 0.0

    closed_rows = [r for r in rows if r["closed"]]
    open_rows = [r for r in rows if not r["closed"]]
    in_vocab = sum(r["answer"] in vocab for r in rows)
    metrics = {
        "overall": accuracy(rows),
        "closed": accuracy(closed_rows),
        "open": accuracy(open_rows),
        "n_test": len(rows),
        "n_closed": len(closed_rows),
        "n_open": len(open_rows),
        "vocab_coverage": in_vocab / len(rows),
    }

    print(f"Checkpoint: {checkpoint} (run: {cfg.run_name}, fusion: {cfg.fusion})")
    print(f"Overall: {metrics['overall']:.4f} (n={metrics['n_test']})")
    print(f"Closed (yes/no): {metrics['closed']:.4f} (n={metrics['n_closed']})")
    print(f"Open: {metrics['open']:.4f} (n={metrics['n_open']})")
    print(f"Độ phủ vocab trên test: {metrics['vocab_coverage']:.1%}"
          f" — trần accuracy khả dĩ")

    # Bảng ví dụ đúng/sai — nguyên liệu phân tích lỗi cho báo cáo
    def show(title, subset):
        print(f"\n--- {title} ---")
        for r in subset[:5]:
            print(f"  Q: {r['question'][:60]:60s} | gt: {r['answer'][:20]:20s}"
                  f" | pred: {r['pred'][:20]}")

    show("Ví dụ dự đoán ĐÚNG", [r for r in rows if r["correct"]])
    show("Ví dụ dự đoán SAI", [r for r in rows if not r["correct"]])

    out_path = Path(cfg.output_dir) / cfg.run_name / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "rows": rows},
                                   indent=2, ensure_ascii=False))
    print(f"\nKết quả chi tiết → {out_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedVQA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
