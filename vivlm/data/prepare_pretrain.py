"""FineWeb2-HQ vie_Latn (parquet) -> train.bin/val.bin uint16.

Chỉ đọc cột "text" (parquet còn cột embeddings rất nặng — bỏ qua).
Tải TỪNG file đến khi đủ token thay vì cố định số file.
"""
import argparse
import os

import numpy as np

REPO = "epfml/FineWeb2-HQ"
SUBSET = "vie_Latn"
N_FILES = 72          # 000_00000.parquet .. (xem HF)


class BinWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "wb")
        self.count = 0

    def write(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(self.f)
        self.count += len(arr)

    def close(self):
        self.f.close()


def tokenize_parquet(parquet_path, tok, writer, eos_id):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=["text"], batch_size=1024):
        texts = batch.column("text").to_pylist()
        for enc in tok.encode_batch(texts):       # song song đa lõi (rayon)
            writer.write(enc.ids + [eos_id])


def main():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="vivlm/data/tokenizer.json")
    ap.add_argument("--out-dir", default="vivlm/data/bin")
    ap.add_argument("--target-tokens", type=float, default=3e9)
    ap.add_argument("--val-tokens", type=float, default=3e7)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<|endoftext|>")
    train = BinWriter(os.path.join(args.out_dir, "train.bin"))
    val = BinWriter(os.path.join(args.out_dir, "val.bin"))

    i = 0
    while train.count < args.target_tokens and i < N_FILES:
        path = hf_hub_download(REPO, f"{SUBSET}/000_{i:05d}.parquet",
                               repo_type="dataset")
        tokenize_parquet(path, tok, train, eos)
        print(f"file {i}: train.bin = {train.count/1e9:.3f}B token")
        i += 1
    # file kế tiếp cho val (không trùng train)
    path = hf_hub_download(REPO, f"{SUBSET}/000_{i:05d}.parquet",
                           repo_type="dataset")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["text"], batch_size=1024):
        if val.count >= args.val_tokens:
            break
        for enc in tok.encode_batch(batch.column("text").to_pylist()):
            val.write(enc.ids + [eos])
    train.close()
    val.close()
    print(f"XONG: train {train.count/1e9:.3f}B, val {val.count/1e6:.1f}M token")


if __name__ == "__main__":
    main()
