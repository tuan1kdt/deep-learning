"""Giai đoạn C: SCST (Self-Critical Sequence Training).

REINFORCE với baseline là chính model decode greedy:
    loss = -(CIDEr(sample) - CIDEr(greedy)) * log P(sample)
Sample tốt hơn greedy -> đẩy xác suất lên; tệ hơn -> đẩy xuống.
Sampling chạy no_grad; logprob lấy lại bằng MỘT forward teacher-forcing.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from vivlm.config import SCSTConfig, SFTConfig, pick_device
from vivlm.data.sft_dataset import preprocess_image
from vivlm.evaluate import _norm


def caption_mask(sampled, eos_id):
    is_eos = (sampled == eos_id).float()
    after = is_eos.cumsum(1) - is_eos                # >0 sau eos đầu tiên
    return (after == 0).float()


def scst_loss(logprobs, mask, advantages):
    return -(advantages * (logprobs * mask).sum(1)).mean()


def sequence_logprobs(vlm, px, prompt_ids, sampled):
    seq = torch.cat([prompt_ids, sampled], dim=1)
    input_ids = seq[:, :-1]
    dummy = torch.zeros_like(input_ids)              # chỉ cần full logits
    logits, _ = vlm(px, input_ids, labels=dummy)
    n_img = logits.size(1) - input_ids.size(1)
    # vị trí fused dự đoán sampled[t]: n_img + len(prompt) - 1 + t
    start = n_img + prompt_ids.size(1) - 1
    logp = F.log_softmax(logits[:, start:start + sampled.size(1), :], dim=-1)
    return logp.gather(2, sampled.unsqueeze(2)).squeeze(2)


def cider_rewards(hyps, refs):
    from pycocoevalcap.cider.cider import Cider
    gts = {i: [_norm(x) for x in r] for i, r in enumerate(refs)}
    res = {i: [_norm(h)] for i, h in enumerate(hyps)}
    _, per_image = Cider().compute_score(gts, res)
    return np.asarray(per_image, dtype=np.float64)


def train_scst(cfg: SCSTConfig, sft_ckpt, device=None, encoder=None,
               tokenizer=None):
    from PIL import Image
    from tokenizers import Tokenizer
    from vivlm.sft import load_vlm, save_vlm
    device = device or pick_device()
    torch.manual_seed(cfg.seed)
    tok = tokenizer or Tokenizer.from_file("vivlm/data/tokenizer.json")
    eos = tok.token_to_id("<|endoftext|>")
    vlm = load_vlm(sft_ckpt, SFTConfig(), device, encoder)
    params = vlm.trainable_parameters("full")
    for p in vlm.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    groups = [json.loads(l) for l in
              open(cfg.refs_jsonl, encoding="utf-8") if l.strip()]
    prompt_ids_1 = tok.encode(f"<|user|> {cfg.caption_prompt} <|assistant|>").ids
    img_root = os.path.dirname(cfg.refs_jsonl)
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    for step in range(cfg.max_steps):
        batch = [groups[i] for i in
                 rng.integers(0, len(groups), cfg.batch_size)]
        px = torch.stack([preprocess_image(
            Image.open(f"{img_root}/{g['image']}")) for g in batch]).to(device)
        prompt = torch.tensor([prompt_ids_1] * len(batch), device=device)

        vlm.eval()
        with torch.no_grad():
            greedy = vlm.generate(px, prompt, cfg.max_new_tokens,
                                  temperature=0.0, eos_id=eos)
            sampled = vlm.generate(px, prompt, cfg.max_new_tokens,
                                   temperature=1.0, eos_id=eos)
        refs = [g["refs"] for g in batch]

        def dec(t):
            ids = []
            for x in t.tolist():
                if x == eos:            # CẮT tại eos đầu tiên — batch>1 sinh
                    break               # tiếp sau eos của ảnh xong sớm (rác)
                ids.append(x)
            return tok.decode(ids).strip()

        r_g = cider_rewards([dec(t) for t in greedy], refs)
        r_s = cider_rewards([dec(t) for t in sampled], refs)
        adv = torch.tensor(r_s - r_g, dtype=torch.float32, device=device)

        vlm.train()
        # pad sampled ngắn hơn max_new bằng eos để thành tensor chữ nhật
        lp = sequence_logprobs(vlm, px, prompt, sampled)
        mask = caption_mask(sampled, eos)
        loss = scst_loss(lp, mask, adv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        print(f"step {step+1}/{cfg.max_steps} r_greedy {r_g.mean():.3f} "
              f"r_sample {r_s.mean():.3f} loss {loss.item():.4f}")
        _log(cfg.log_csv, [step + 1, f"{r_g.mean():.4f}",
                           f"{r_s.mean():.4f}", f"{loss.item():.4f}"])
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.max_steps:
            save_vlm(os.path.join(cfg.out_dir, "scst.pt"), vlm, step + 1)


def _log(csv_path, row):
    new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "reward_greedy", "reward_sample", "loss"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-ckpt", default="vivlm/checkpoints/sft/full.pt")
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    cfg = SCSTConfig()
    if args.max_steps:
        cfg.max_steps = args.max_steps
    train_scst(cfg, args.sft_ckpt)


if __name__ == "__main__":
    main()
