"""Đánh giá caption trên test: BLEU-1..4 (nltk) + CIDEr (pycocoevalcap).

CIDEr dùng module thuần Python của pycocoevalcap, tự tokenize bằng
final.data.vocab.tokenize thay vì PTBTokenizer (né dependency Java) — chấp
nhận lệch nhỏ so với số liệu chuẩn hóa COCO, ghi chú trong báo cáo.

Chạy: .venv/bin/python -m final.evaluate --checkpoint final/checkpoints/lstm.pt
"""
import argparse
import json
from pathlib import Path

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from pycocoevalcap.cider.cider import Cider

from final.config import Config, pick_device
from final.data.dataset import load_eval_data
from final.data.vocab import Vocab, tokenize
from final.models.caption_model import build_model
from final.models.decoding import beam_search, greedy_decode


def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = dict(ckpt["config"])
    saved["beam_sizes"] = tuple(saved.get("beam_sizes", (3, 5)))
    cfg = Config(**saved)
    vocab = Vocab.load(cfg.vocab_path)
    assert len(vocab) == ckpt["vocab_size"], "vocab trên đĩa khác lúc train"
    model = build_model(cfg, len(vocab)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, vocab


def compute_metrics(hyps: list[str], refs: list[list[str]]) -> dict:
    """hyps: câu dự đoán; refs: 5 câu tham chiếu thô mỗi ảnh. Tokenize chung
    một kiểu cho cả hai phía để so sánh công bằng."""
    hyp_tok = [tokenize(h) for h in hyps]
    ref_tok = [[tokenize(r) for r in group] for group in refs]
    smooth = SmoothingFunction().method1
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0),
               (1 / 3, 1 / 3, 1 / 3, 0), (0.25, 0.25, 0.25, 0.25)]
    bleu = {f"bleu{i + 1}": corpus_bleu(ref_tok, hyp_tok, weights=w,
                                        smoothing_function=smooth)
            for i, w in enumerate(weights)}
    gts = {i: [" ".join(toks) for toks in group] for i, group in enumerate(ref_tok)}
    res = {i: [" ".join(hyp_tok[i])] for i in range(len(hyp_tok))}
    cider_score, _ = Cider().compute_score(gts, res)
    return {**bleu, "cider": float(cider_score)}


def generate(model, feats, mode: str, cfg: Config, vocab: Vocab,
             device, batch_size: int = 256) -> list[str]:
    if mode == "greedy":
        out = []
        for i in range(0, len(feats), batch_size):
            seqs, _ = greedy_decode(model, feats[i:i + batch_size].to(device),
                                    cfg.max_len)
            out += [vocab.decode(s) for s in seqs]
        return out
    assert mode.startswith("beam")
    k = int(mode.removeprefix("beam"))
    out = []
    for i in range(len(feats)):
        ids = beam_search(model, feats[i:i + 1].to(device), k, cfg.max_len)
        out.append(vocab.decode(ids))
        if (i + 1) % 200 == 0:
            print(f"  {mode}: {i + 1}/{len(feats)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modes", default="greedy,beam3,beam5")
    parser.add_argument("--limit", type=int, default=0,
                        help="chỉ chấm N ảnh đầu (0 = cả 1000, dùng khi smoke)")
    args = parser.parse_args()

    device = pick_device()
    model, cfg, vocab = load_model_from_checkpoint(args.checkpoint, device)
    feats, refs = load_eval_data(cfg, "test")
    if args.limit:
        feats, refs = feats[:args.limit], refs[:args.limit]
    print(f"Đánh giá {cfg.run_name} trên {len(feats)} ảnh test")

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["| mode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr |",
            "|---|---|---|---|---|---|"]
    for mode in args.modes.split(","):
        hyps = generate(model, feats, mode, cfg, vocab, device)
        (out_dir / f"hyps_{mode}.json").write_text(
            json.dumps(hyps, ensure_ascii=False, indent=2))
        m = compute_metrics(hyps, refs)
        rows.append(f"| {mode} | {m['bleu1']:.3f} | {m['bleu2']:.3f} "
                    f"| {m['bleu3']:.3f} | {m['bleu4']:.3f} | {m['cider']:.3f} |")
        print(rows[-1])
    table = "\n".join(rows)
    (out_dir / "eval.md").write_text(f"# Eval {cfg.run_name}\n\n{table}\n")
    print(f"\n{table}\n→ {out_dir / 'eval.md'}")


if __name__ == "__main__":
    main()
