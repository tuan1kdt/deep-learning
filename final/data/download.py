"""Tải Flickr8k/Flickr30k từ HF về đĩa + trích caption ra JSON nhẹ.

Nguồn Flickr8k: jxie/flickr8k — mirror nhiều lượt tải nhất, ĐÃ chia sẵn
Karpathy split (train 6000 / validation 1000 / test 1000, mỗi ảnh 5 caption,
5 cột caption_0..4). Nguồn Flickr30k: nlphuji/flickr30k — MỘT split duy nhất
với cột `split` (train/val/test, Karpathy) và cột `caption` là list 5 câu;
ta tự tách thành DatasetDict 3 split cùng khuôn với Flickr8k.

Chạy: .venv/bin/python -m final.data.download [--dataset flickr30k]
Idempotent: đã có trên đĩa thì chỉ in thống kê rồi thoát.
"""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from final.config import Config

SPLITS = ("train", "validation", "test")
CAPTION_KEYS = tuple(f"caption_{i}" for i in range(5))


def load_captioning_dataset(cfg: Config) -> DatasetDict:
    """Trả về DatasetDict 3 split; tải + save_to_disk nếu chưa có."""
    ds_dir = Path(cfg.dataset_dir)
    if (ds_dir / "dataset_dict.json").exists():
        return load_from_disk(str(ds_dir))
    if cfg.dataset == "flickr30k":
        # nlphuji/flickr30k là dataset dạng script — datasets>=3 bỏ hỗ trợ,
        # nên đọc thẳng bản parquet HF tự convert (revision refs/convert/parquet;
        # toàn bộ nằm trong config TEST / split test, builder parquet đặt tên
        # split mặc định là "train").
        raw = load_dataset(
            "parquet",
            data_files="hf://datasets/nlphuji/flickr30k@refs/convert/parquet"
                       "/TEST/test/*.parquet",
            split="train",
        )
        # input_columns=["split"] để filter chỉ đọc cột text, không decode ảnh
        ds = DatasetDict({
            name: raw.filter(lambda s: s == src, input_columns=["split"])
            for name, src in
            (("train", "train"), ("validation", "val"), ("test", "test"))
        })
    else:
        ds = load_dataset(cfg.hf_dataset)
    ds.save_to_disk(str(ds_dir))
    return ds


# Giữ tên cũ cho tương thích các import hiện có
load_flickr8k = load_captioning_dataset


def extract_captions(rows) -> list[list[str]]:
    """[[5 caption thô của ảnh 0], [5 caption của ảnh 1], ...] — index là
    thứ tự ảnh trong split, mọi artifact sau (features, refs) đều căn theo đó.
    Hai schema: cột `caption` là list (flickr30k) hoặc 5 cột caption_i (flickr8k)."""
    if len(rows) and "caption" in rows[0]:
        return [list(row["caption"]) for row in rows]
    return [[row[k] for k in CAPTION_KEYS] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="flickr8k",
                        choices=["flickr8k", "flickr30k"])
    args = parser.parse_args()
    cfg = Config(dataset=args.dataset)
    ds = load_captioning_dataset(cfg)
    for split in SPLITS:
        out = cfg.captions_path(split)
        if not out.exists():
            # .select_columns tránh decode cột ảnh (nặng) khi chỉ cần text
            cap_cols = (["caption"] if "caption" in ds[split].column_names
                        else list(CAPTION_KEYS))
            caps = extract_captions(ds[split].select_columns(cap_cols))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(caps, ensure_ascii=False))
        n = len(json.loads(out.read_text()))
        print(f"{split:10s}: {len(ds[split]):5d} ảnh | {n:5d} nhóm caption → {out}")


if __name__ == "__main__":
    main()
