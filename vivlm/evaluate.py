"""Đánh giá: perplexity (text), CIDEr/BLEU-4 (caption/VQA), beam search.

Không dùng PTBTokenizer của pycocoevalcap (cần Java) — tiếng Việt chuẩn hóa
lowercase + tách whitespace là đủ và nhất quán giữa các hệ thống so sánh.
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from vivlm.config import SFTConfig, pick_device
from vivlm.data.loader import TokenBin
from vivlm.data.sft_dataset import preprocess_image


def perplexity(model, bin_path, context, device, iters=200):
    tb = TokenBin(bin_path)
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(iters):
            s = (i * context) % (len(tb) - context - 1)
            x = torch.from_numpy(
                tb.data[s:s + context].astype("int64"))[None].to(device)
            y = torch.from_numpy(
                tb.data[s + 1:s + 1 + context].astype("int64"))[None].to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


def bits_per_char(model, tok, bin_path, context, device, iters=100):
    """Bits/ký tự — thước đo so sánh được giữa các tokenizer khác nhau."""
    tb = TokenBin(bin_path)
    model.eval()
    total_nll, total_chars = 0.0, 0
    with torch.no_grad():
        for i in range(iters):
            s = (i * context) % (len(tb) - context - 1)
            x = torch.from_numpy(
                tb.data[s:s + context].astype("int64"))[None].to(device)
            y = torch.from_numpy(
                tb.data[s + 1:s + 1 + context].astype("int64"))[None].to(device)
            _, loss = model(x, y)
            text = tok.decode(tb.data[s + 1:s + 1 + context]
                              .astype("int64").tolist())
            total_nll += loss.item() * context
            total_chars += max(1, len(text))
    return total_nll / math.log(2) / total_chars


def baseline_bits_per_char(text, device, name="NlpHUST/gpt2-vietnamese",
                           window=512):
    """bpc của baseline HF trên CÙNG văn bản (transformers chỉ để eval)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    btok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    ids = btok(text, return_tensors="pt").input_ids[0]
    total_nll, total_chars = 0.0, len(text)
    with torch.no_grad():
        for s in range(0, len(ids) - 1, window):
            chunk = ids[s:s + window + 1][None].to(device)
            out = m(chunk[:, :-1], labels=chunk[:, 1:])
            total_nll += out.loss.item() * (chunk.size(1) - 1)
    return total_nll / math.log(2) / total_chars


def _norm(s):
    return " ".join(s.lower().split())


def score_captions(gts, res):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    g = {k: [_norm(x) for x in v] for k, v in gts.items()}
    r = {k: [_norm(v)] for k, v in res.items()}
    bleu, _ = Bleu(4).compute_score(g, r)
    cider, _ = Cider().compute_score(g, r)
    return {"Bleu_4": bleu[3], "CIDEr": cider}


@torch.no_grad()
def beam_generate(vlm, pixel_values, prompt_ids, beam_size=3, max_new=40,
                  eos_id=0, length_penalty=0.7):
    """Beam search với length normalization (Wu et al. 2016 dạng đơn giản):
    điểm mỗi beam chia cho (số token sinh)^length_penalty, tránh thiên vị
    câu ngắn khi beam_size > 1 (log-prob cộng dồn luôn <= 0 nên câu dài hơn
    bị phạt nặng hơn nếu không chuẩn hóa)."""
    assert pixel_values.size(0) == 1, "beam search chạy từng ảnh"
    P = prompt_ids.size(1)

    def norm_score(b):
        return b[1] / (max(1, b[0].size(1) - P) ** length_penalty)

    beams = [(prompt_ids, 0.0, False)]
    for _ in range(max_new):
        cand = []
        for ids, lp, done in beams:
            if done:
                cand.append((ids, lp, True))
                continue
            x, _ = vlm._fuse(pixel_values, ids)
            logits, _ = vlm.gpt.forward_from_embeds(x)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)[0]
            top = torch.topk(logp, beam_size)
            for v, i in zip(top.values, top.indices):
                nids = torch.cat([ids, i.view(1, 1)], dim=1)
                cand.append((nids, lp + v.item(), i.item() == eos_id))
        beams = sorted(cand, key=lambda b: -norm_score(b))[:beam_size]
        if all(b[2] for b in beams):
            break
    return max(beams, key=norm_score)[0][:, P:]


def eval_captions(vlm, tok, jsonl_path, img_root, device, prompt=None,
                  beam=0, max_new=40, limit=None):
    from PIL import Image
    eos = tok.token_to_id("<|endoftext|>")
    recs = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")
            if l.strip()][:limit]
    gts, res, preds = {}, {}, []
    vlm.eval()
    for i, r in enumerate(recs):
        with Image.open(f"{img_root}/{r['image']}") as im:
            px = preprocess_image(im).unsqueeze(0).to(device)
        p = prompt if prompt is not None else r["prompt"]
        ids = torch.tensor([tok.encode(f"<|user|> {p} <|assistant|>").ids],
                           device=device)
        if beam > 0:
            out = beam_generate(vlm, px, ids, beam, max_new, eos)
        else:
            out = vlm.generate(px, ids, max_new, temperature=0.0, eos_id=eos)
        hyp = tok.decode([t for t in out[0].tolist() if t != eos]).strip()
        gts[i], res[i] = r["refs"], hyp
        preds.append({"image": r["image"], "hyp": hyp, "refs": r["refs"]})
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(recs)}")
    with open(jsonl_path + ".pred.jsonl", "w", encoding="utf-8") as f:
        for p_ in preds:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    return score_captions(gts, res)


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ppl", "bpc", "caption", "vqa"],
                    required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="vivlm/data/tokenizer.json")
    ap.add_argument("--jsonl")
    ap.add_argument("--img-root", default="vivlm/data/sft")
    ap.add_argument("--val-bin", default="vivlm/data/bin/val.bin")
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    device = pick_device()
    tok = Tokenizer.from_file(args.tokenizer)
    if args.mode == "ppl":
        from vivlm.sample import load_model
        m = load_model(args.ckpt, device)
        print(f"PPL = {perplexity(m, args.val_bin, m.cfg.context, device):.2f}")
        return
    if args.mode == "bpc":
        from vivlm.sample import load_model
        m = load_model(args.ckpt, device)
        ours = bits_per_char(m, tok, args.val_bin, m.cfg.context, device)
        tb = TokenBin(args.val_bin)
        text = tok.decode(tb.data[:100_000].astype("int64").tolist())
        base = baseline_bits_per_char(text, device)
        print(f"bits/char: ours = {ours:.3f} | NlpHUST/gpt2-vietnamese = {base:.3f}")
        return
    from vivlm.sft import load_vlm
    vlm = load_vlm(args.ckpt, SFTConfig(), device)
    prompt = "Mô tả bức ảnh." if args.mode == "caption" else None
    scores = eval_captions(vlm, tok, args.jsonl, args.img_root, device,
                           prompt=prompt, beam=args.beam, limit=args.limit)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
