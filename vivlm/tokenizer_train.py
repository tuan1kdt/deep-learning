"""Train byte-level BPE cho tiếng Việt + so sánh fertility với PhoGPT/GPT-2.

Byte-level: không bao giờ <unk> — mọi ký tự UTF-8 (kể cả dấu tiếng Việt) phân rã
được về byte. Fertility = token/từ: càng thấp, tokenizer càng "hiểu" ngôn ngữ.
"""
import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from vivlm.config import SPECIAL_TOKENS


def train_bpe(files, out_path, vocab_size=20480):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,      # chiếm id 0..3
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train(files, trainer)
    tok.save(out_path)
    return tok


def extract_text(parquet_paths, out_txt, max_bytes):
    import pyarrow.parquet as pq
    written = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for path in parquet_paths:
            for batch in pq.ParquetFile(path).iter_batches(
                    columns=["text"], batch_size=1024):
                for t in batch.column("text").to_pylist():
                    line = t.replace("\n", " ") + "\n"
                    f.write(line)
                    written += len(line.encode("utf-8"))
                    if written >= max_bytes:
                        return
            print(f"đã ghi {written/1e9:.2f} GB sau {path}")


def fertility(tok, text):
    return len(tok.encode(text).ids) / len(text.split())


def compare_fertility(ours_path, sample_text):
    from huggingface_hub import hf_hub_download
    rows = [("ours (20480)", Tokenizer.from_file(ours_path))]
    for label, repo in [("PhoGPT (20480)", "vinai/PhoGPT-4B"),
                        ("GPT-2 en (50257)", "openai-community/gpt2")]:
        path = hf_hub_download(repo, "tokenizer.json")
        rows.append((label, Tokenizer.from_file(path)))
    print(f"{'tokenizer':<20} {'fertility (token/từ)':>22}")
    for label, tok in rows:
        print(f"{label:<20} {fertility(tok, sample_text):>22.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", help="file parquet nguồn")
    ap.add_argument("--out", default="vivlm/data/tokenizer.json")
    ap.add_argument("--sample-txt", default="vivlm/data/tok_sample.txt")
    ap.add_argument("--max-bytes", type=float, default=2e9)
    ap.add_argument("--vocab-size", type=int, default=20480)
    ap.add_argument("--compare", action="store_true",
                    help="in bảng fertility (cần mạng để tải PhoGPT/GPT-2)")
    args = ap.parse_args()
    if args.parquet:
        extract_text(args.parquet, args.sample_txt, int(args.max_bytes))
        train_bpe([args.sample_txt], args.out, args.vocab_size)
        print(f"đã lưu {args.out}")
    if args.compare:
        sample = open(args.sample_txt, encoding="utf-8").read(2_000_000)
        compare_fertility(args.out, sample)


if __name__ == "__main__":
    main()
