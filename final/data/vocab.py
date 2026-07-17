"""Vocab mức từ cho caption: chuẩn hóa đơn giản, ngưỡng tần suất, id cố định.

Chọn mức từ (không BPE) vì Flickr8k nhỏ (~8k từ ≥5 lần) và mức từ dễ giảng
giải trong báo cáo — mỗi token là một từ nhìn thấy được trên attention heatmap.

Chạy: .venv/bin/python -m final.data.vocab   (build từ captions_train.json)
"""
import json
import re
from collections import Counter
from pathlib import Path

from final.config import Config

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# \w khớp cả chữ có dấu (tiếng Việt trong test) nhờ Unicode mặc định của re
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """lowercase → bỏ ký tự không phải chữ/số → tách theo khoảng trắng."""
    return _NON_WORD.sub(" ", text.lower()).split()


class Vocab:
    def __init__(self, word2id: dict[str, int]):
        self.word2id = word2id
        self.id2word = {i: w for w, i in word2id.items()}

    def __len__(self) -> int:
        return len(self.word2id)

    def encode(self, text: str, max_words: int) -> list[int]:
        """Chỉ id của từ (không bos/eos) — dataset tự ghép khung câu."""
        words = tokenize(text)[:max_words]
        return [self.word2id.get(w, UNK_ID) for w in words]

    def decode(self, ids) -> str:
        """Dừng ở EOS, bỏ pad/bos/unk — dùng cho in mẫu và evaluate."""
        words = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            words.append(self.id2word.get(i, "<unk>"))
        return " ".join(words)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.word2id, ensure_ascii=False))

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


def build_vocab(all_captions: list[str], min_freq: int) -> Vocab:
    counter = Counter(w for cap in all_captions for w in tokenize(cap))
    word2id = {w: i for i, w in enumerate(SPECIALS)}
    # sort theo (-tần suất, từ) để vocab ổn định giữa các lần build
    for w, c in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            word2id[w] = len(word2id)
    return Vocab(word2id)


def main() -> None:
    cfg = Config()
    caps_nested = json.loads(cfg.captions_path("train").read_text())
    flat = [c for group in caps_nested for c in group]
    vocab = build_vocab(flat, cfg.min_word_freq)
    vocab.save(cfg.vocab_path)

    lengths = [len(tokenize(c)) for c in flat]
    cover = sum(l <= cfg.max_words for l in lengths) / len(lengths)
    n_unk = sum(1 for c in flat for i in vocab.encode(c, cfg.max_words) if i == UNK_ID)
    n_tok = sum(min(l, cfg.max_words) for l in lengths)
    print(f"Vocab: {len(vocab)} từ (min_freq={cfg.min_word_freq})")
    print(f"Caption ≤ {cfg.max_words} từ: {cover:.1%} | tỷ lệ UNK: {n_unk / n_tok:.2%}")
    print(f"Đã lưu → {cfg.vocab_path}")


if __name__ == "__main__":
    main()
