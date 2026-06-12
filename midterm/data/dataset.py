"""Dataset PyTorch cho VQA-RAD.

Mỗi mẫu: ảnh → tensor chuẩn hóa ImageNet, câu hỏi → token BERT (pad/truncate
max_len), đáp án → chỉ số class trong answer vocab.
"""
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from torchvision import transforms

from midterm.data.vocab import normalize_answer

# mean/std ImageNet — bắt buộc đúng bộ số này vì ResNet được pretrain với nó
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(cfg, train: bool):
    """Train: random resized crop nhẹ (scale 0.9–1.0) làm augmentation, tắt được
    qua cfg.augment. KHÔNG horizontal flip — ảnh y khoa có tính trái/phải
    (tim nằm bên trái, gan bên phải...). Eval: chỉ resize."""
    if train and cfg.augment:
        resize = transforms.RandomResizedCrop(
            cfg.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0))
    else:
        resize = transforms.Resize((cfg.image_size, cfg.image_size))
    return transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class VQARadDataset(Dataset):
    def __init__(self, hf_split, tokenizer, vocab, cfg, train: bool):
        self.ds = hf_split
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.cfg = cfg
        self.transform = build_transforms(cfg, train)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = self.transform(row["image"].convert("RGB"))
        tokens = self.tokenizer(
            row["question"],
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_question_len,
            return_tensors="pt",
        )
        answer = normalize_answer(row["answer"])
        # Đáp án ngoài vocab (chỉ xảy ra ở test) → label -1: argmax không bao giờ
        # bằng -1 nên evaluate tự động tính là sai — đúng tinh thần "minh bạch".
        label = self.vocab.get(answer, -1)
        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "question": row["question"],  # giữ string cho bảng ví dụ ở evaluate
            "answer": answer,
        }


def load_splits(cfg):
    """Trả về (train, val, test) dạng HF dataset.

    Val tách 10% từ train THEO QA PAIR với seed cố định — lựa chọn có chủ đích:
    VQA-RAD chỉ có 313 ảnh duy nhất cho 1.793 QA pairs (~5,7 câu hỏi/ảnh), và
    202/203 ảnh test cũng xuất hiện trong train — split chính thức vốn chia theo
    câu hỏi chứ không theo ảnh. Val theo QA pair vì vậy khớp đúng "điều kiện thi"
    của test: ảnh đã thấy, câu hỏi mới. (Chi tiết: spec mục 4.3.)
    """
    ds = load_from_disk(cfg.data_dir)
    split = ds["train"].train_test_split(test_size=cfg.val_fraction, seed=cfg.seed)
    return split["train"], split["test"], ds["test"]
