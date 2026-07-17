"""Tải Flickr8k từ HF về đĩa + trích caption ra JSON nhẹ.

Nguồn: jxie/flickr8k — mirror nhiều lượt tải nhất, ĐÃ chia sẵn Karpathy split
(train 6000 / validation 1000 / test 1000, mỗi ảnh 5 caption). Nhờ vậy không
phải tự xử lý dataset_flickr8k.json như phương án fallback trong spec.

Chạy: .venv/bin/python -m final.data.download
Idempotent: đã có trên đĩa thì chỉ in thống kê rồi thoát.
"""
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

from final.config import Config

SPLITS = ("train", "validation", "test")
CAPTION_KEYS = tuple(f"caption_{i}" for i in range(5))


def load_flickr8k(cfg: Config) -> DatasetDict:
    """Trả về DatasetDict 3 split; tải + save_to_disk nếu chưa có."""
    ds_dir = Path(cfg.dataset_dir)
    if (ds_dir / "dataset_dict.json").exists():
        return load_from_disk(str(ds_dir))
    ds = load_dataset(cfg.hf_dataset)
    ds.save_to_disk(str(ds_dir))
    return ds


def extract_captions(rows) -> list[list[str]]:
    """[[5 caption thô của ảnh 0], [5 caption của ảnh 1], ...] — index là
    thứ tự ảnh trong split, mọi artifact sau (features, refs) đều căn theo đó."""
    return [[row[k] for k in CAPTION_KEYS] for row in rows]


def main() -> None:
    cfg = Config()
    ds = load_flickr8k(cfg)
    for split in SPLITS:
        out = cfg.captions_path(split)
        if not out.exists():
            # .select_columns tránh decode cột ảnh (nặng) khi chỉ cần text
            caps = extract_captions(ds[split].select_columns(list(CAPTION_KEYS)))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(caps, ensure_ascii=False))
        n = len(json.loads(out.read_text()))
        print(f"{split:10s}: {len(ds[split]):5d} ảnh | {n:5d} nhóm caption → {out}")


if __name__ == "__main__":
    main()
