"""Sinh văn bản từ checkpoint pretrain (demo + kiểm tra định tính)."""
import argparse

import torch

from vivlm.config import GPTConfig, pick_device
from vivlm.models.gpt import GPT
from vivlm.pretrain import load_ckpt


def load_model(ckpt_path, device):
    ck = load_ckpt(ckpt_path, device)
    model = GPT(GPTConfig(**ck["gpt_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def generate_text(model, tok, prompt, max_new=200, temperature=0.8,
                  top_p=0.9, device="cpu"):
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, temperature=temperature, top_p=top_p,
                         eos_id=tok.token_to_id("<|endoftext|>"))
    return tok.decode(out[0].tolist())


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="vivlm/checkpoints/pretrain/latest.pt")
    ap.add_argument("--tokenizer", default="vivlm/data/tokenizer.json")
    ap.add_argument("--prompt", default="Hà Nội là")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    device = pick_device()
    model = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    print(generate_text(model, tok, args.prompt, args.max_new,
                        temperature=0.0 if args.greedy else 0.8,
                        top_p=None if args.greedy else 0.9, device=device))


if __name__ == "__main__":
    main()
