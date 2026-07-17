"""SCST (Self-Critical Sequence Training) — RL fine-tune sau SFT.

Ý tưởng: REINFORCE với baseline là chính model ở chế độ greedy:
    loss = -(CIDEr(sample) - CIDEr(greedy)) * logprob(sample)
Sample tốt hơn greedy → advantage dương → tăng xác suất chuỗi đó, và ngược
lại. Không cần train value network — baseline "tự phê bình" là điểm greedy.

Đúng mạch "SFT → RL" giảng viên khoanh trên bảng. Reward CIDEr tính theo
batch (IDF trên batch — nhiễu hơn IDF toàn corpus, chấp nhận và ghi chú).

Chạy: .venv/bin/python -m final.scst --checkpoint final/checkpoints/transformer.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from final.config import pick_device
from final.data.dataset import load_eval_data
from final.data.vocab import BOS_ID, EOS_ID, tokenize
from final.evaluate import load_model_from_checkpoint
from final.models.decoding import greedy_decode
from pycocoevalcap.cider.cider import Cider


def cider_rewards(hyps: list[str], refs: list[list[str]]) -> np.ndarray:
    """CIDEr từng câu (không phải trung bình corpus) làm reward."""
    gts = {i: [" ".join(tokenize(r)) for r in group]
           for i, group in enumerate(refs)}
    res = {i: [" ".join(tokenize(h))] for i, h in enumerate(hyps)}
    _, per_image = Cider().compute_score(gts, res)
    return np.asarray(per_image, dtype=np.float32)


def sample_decode(model, feats, max_len: int):
    """Sample multinomial từng bước, giữ graph để backprop tổng log-prob.
    Trả (chuỗi id đã cắt bos/eos, tổng log-prob các token đã sinh)."""
    B = feats.size(0)
    device = feats.device
    cap = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
    logp_sum = torch.zeros(B, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        logits, _ = model(feats, cap)
        dist = torch.distributions.Categorical(logits=logits[:, -1])
        nxt = dist.sample()
        # token sau khi câu đã kết thúc không đóng góp log-prob
        logp_sum = logp_sum + dist.log_prob(nxt) * (~done)
        done = done | (nxt == EOS_ID)
        cap = torch.cat([cap, nxt.unsqueeze(1)], dim=1)
        if done.all():
            break
    seqs = []
    for row in cap.tolist():
        out = []
        for i in row[1:]:
            if i == EOS_ID:
                break
            out.append(i)
        seqs.append(out)
    return seqs, logp_sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="256 ảnh, 1 epoch")
    args = parser.parse_args()

    device = pick_device()
    model, cfg, vocab = load_model_from_checkpoint(args.checkpoint, device)
    feats, refs = load_eval_data(cfg, "train")
    if args.smoke:
        feats, refs = feats[:256], refs[:256]
        args.epochs = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=cfg.weight_decay)

    run_name = f"{cfg.run_name}_scst"
    ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"
    history = {"reward_sample": [], "reward_greedy": []}

    n = feats.size(0)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        ep_rs, ep_rg = [], []
        for i0 in range(0, n, args.batch_size):
            idx = perm[i0:i0 + args.batch_size]
            f = feats[idx].to(device)
            r = [refs[j] for j in idx.tolist()]

            model.eval()   # baseline greedy không dropout
            with torch.no_grad():
                greedy_ids, _ = greedy_decode(model, f, cfg.max_len)
            model.train()
            sample_ids, logps = sample_decode(model, f, cfg.max_len)

            r_greedy = cider_rewards([vocab.decode(s) for s in greedy_ids], r)
            r_sample = cider_rewards([vocab.decode(s) for s in sample_ids], r)
            advantage = torch.tensor(r_sample - r_greedy, device=device)

            optimizer.zero_grad()
            loss = -(advantage * logps).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ep_rs.append(r_sample.mean())
            ep_rg.append(r_greedy.mean())

        history["reward_sample"].append(float(np.mean(ep_rs)))
        history["reward_greedy"].append(float(np.mean(ep_rg)))
        print(f"Epoch {epoch} | CIDEr sample {np.mean(ep_rs):.3f}"
              f" | greedy {np.mean(ep_rg):.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": {**cfg.to_dict(), "run_name": run_name},
        "vocab_size": len(vocab),
        "epoch": args.epochs,
        "val_loss": -1.0,   # RL không tối ưu CE nữa; đánh giá bằng evaluate.py
    }, ckpt_path)
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"→ {ckpt_path}\nĐánh giá: .venv/bin/python -m final.evaluate"
          f" --checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
