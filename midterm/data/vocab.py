"""Xây answer vocabulary từ train split của VQA-RAD.

Bài toán được mô hình hóa thành classification: mỗi đáp án duy nhất (sau chuẩn
hóa) là một class — KHÔNG có token <unk>: mọi đáp án train đều nằm trong vocab
theo cách build, nên <unk> sẽ là class chết không bao giờ làm target. Đáp án
test ngoài vocab thì model chắc chắn sai; ta in độ phủ để báo cáo minh bạch.

Chạy: python -m midterm.data.vocab
"""
import json
import re
from pathlib import Path

from datasets import load_from_disk

from midterm.config import Config


def normalize_answer(answer: str) -> str:
    """Chuẩn hóa đáp án: lowercase, bỏ khoảng trắng và dấu câu thừa hai đầu,
    gộp khoảng trắng liên tiếp. 'Yes.' / ' yes ' / 'YES' đều thành 'yes'."""
    s = answer.strip().lower()
    s = s.strip(".,;:!? ")
    s = re.sub(r"\s+", " ", s)
    return s


def build_vocab(data_dir: str) -> dict:
    """Mapping answer → index, build từ train split duy nhất.

    sorted() để vocab ổn định giữa các lần build (set không có thứ tự cố định).
    """
    ds = load_from_disk(data_dir)
    answers = sorted({normalize_answer(a) for a in ds["train"]["answer"]})
    return {answer: idx for idx, answer in enumerate(answers)}


def save_vocab(vocab: dict, path: str) -> None:
    Path(path).write_text(json.dumps(vocab, indent=2, ensure_ascii=False))


def load_vocab(path: str) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    cfg = Config()
    vocab = build_vocab(cfg.data_dir)
    save_vocab(vocab, cfg.vocab_path)
    print(f"Vocab: {len(vocab)} class → {cfg.vocab_path}")

    # Độ phủ vocab trên test = trần accuracy khả dĩ (đáp án ngoài vocab chắc chắn sai)
    ds = load_from_disk(cfg.data_dir)
    test_answers = [normalize_answer(a) for a in ds["test"]["answer"]]
    covered = sum(a in vocab for a in test_answers)
    print(f"Độ phủ vocab trên test: {covered}/{len(test_answers)}"
          f" = {100 * covered / len(test_answers):.1f}%")
