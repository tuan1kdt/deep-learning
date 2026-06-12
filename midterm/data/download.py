"""Tải dataset VQA-RAD từ HuggingFace Hub về đĩa (idempotent).

Chạy lại an toàn: nếu dataset đã có trên đĩa thì bỏ qua. Trên Colab chỉ cần
chạy đúng lệnh này (~34MB):

    python -m midterm.data.download
"""
from pathlib import Path

from datasets import load_dataset

from midterm.config import Config


def download(data_dir: str) -> None:
    target = Path(data_dir)
    if (target / "dataset_dict.json").exists():
        print(f"Dataset đã có tại {target} — bỏ qua download.")
        return
    print("Đang tải flaviagiammarino/vqa-rad từ HuggingFace Hub (~34MB)...")
    ds = load_dataset("flaviagiammarino/vqa-rad")
    ds.save_to_disk(str(target))
    print(f"Đã lưu vào {target}: train={len(ds['train'])}, test={len(ds['test'])}")


if __name__ == "__main__":
    download(Config().data_dir)
