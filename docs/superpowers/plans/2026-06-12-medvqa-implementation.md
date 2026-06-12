# MedVQA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ MedVQA hoàn chỉnh trên VQA-RAD (dual-encoder frozen + 3 fusion hoán đổi qua config) gồm data pipeline, model, train/evaluate/demo CLI và notebook Colab, theo spec `docs/superpowers/specs/2026-06-11-medvqa-design.md`.

**Architecture:** ResNet-50 (frozen, BN luôn eval) và BERT-base (frozen, mean-pooling) chiếu về 768-d; 3 module fusion cùng interface `(v_img, img_map, v_txt) → (fused, attn|None)`; MLP head phân loại 429 đáp án. Mọi lựa chọn thí nghiệm đi qua `config.py`. Train trên Colab, smoke test local (MPS).

**Tech Stack:** Python 3.14 (local) / 3.12 (Colab), torch 2.12, torchvision 0.27, transformers 5.10, datasets 5.0, matplotlib (Agg).

**Lưu ý về verification thay cho pytest TDD:** Repo này theo CLAUDE.md là nhật ký học tập — script tự chứa kiểu tutorial, không có hạ tầng test. Thay vì pytest, mỗi task có bước verify chạy code thật (lệnh chính xác + output kỳ vọng): shape check qua `python -c`, smoke run end-to-end qua cờ `--smoke`. Không claim hoàn thành task khi bước verify chưa pass.

**Fact đã xác minh trước (ghi cứng vào expected output):**
- Dataset trên đĩa: train 1.793 / test 451; val split 10% seed 42 → 1.613/180.
- Vocab sau chuẩn hóa đầy đủ: **429 class**; coverage test **334/451 = 74,1%**.
- `.gitignore` đã cover `midterm/data/vqa_rad/`, `answer_vocab.json`, `checkpoints/`, `outputs/`, `*.png` — không cần sửa.
- Remote: `https://github.com/tuan1kdt/deep-learning.git` (dùng trong notebook).
- Mọi lệnh chạy từ repo root, venv đã activate: `source .venv/bin/activate`.

---

### Task 1: Scaffold — requirements.txt, package init, config.py

**Files:**
- Create: `midterm/requirements.txt`
- Create: `midterm/__init__.py`
- Create: `midterm/data/__init__.py`
- Create: `midterm/models/__init__.py`
- Create: `midterm/config.py`

- [ ] **Step 1: Viết `midterm/requirements.txt`** (cho Colab Python 3.12 — pin lỏng, API dùng đều ổn định giữa các version)

```
torch>=2.5
torchvision>=0.20
transformers>=4.45
datasets>=3.0
matplotlib>=3.8
```

- [ ] **Step 2: Tạo 3 file `__init__.py` rỗng** (một docstring ngắn mỗi file)

`midterm/__init__.py`:
```python
"""Đồ án giữa kỳ: Medical VQA trên VQA-RAD."""
```

`midterm/data/__init__.py`:
```python
"""Data pipeline: download, vocab, dataset."""
```

`midterm/models/__init__.py`:
```python
"""Các thành phần model: encoder, fusion, head."""
```

- [ ] **Step 3: Viết `midterm/config.py`**

```python
"""Cấu hình trung tâm cho đồ án MedVQA.

Mọi lựa chọn thí nghiệm (fusion, text pooling, unfreeze...) đều đi qua dataclass
Config — đổi thí nghiệm = đổi config/CLI flag, không sửa code model hay training.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

# Thư mục gốc của package midterm/ — mọi đường dẫn artifact neo theo đây
# để chạy được từ bất kỳ working directory nào (local lẫn Colab).
MIDTERM_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    # ----- Data -----
    data_dir: str = str(MIDTERM_DIR / "data" / "vqa_rad")
    vocab_path: str = str(MIDTERM_DIR / "data" / "answer_vocab.json")
    image_size: int = 224
    max_question_len: int = 32
    val_fraction: float = 0.10  # tách 10% train làm validation (theo QA pair)
    augment: bool = True        # random resized crop nhẹ, chỉ áp dụng cho train

    # ----- Model -----
    d_model: int = 768          # chiều chung sau projection (= chiều ẩn BERT)
    fusion: str = "concat"      # concat | hadamard | cross_attention
    text_model_name: str = "bert-base-uncased"
    text_pool: str = "mean"     # mean | cls — BERT freeze: mean tốt hơn CLS
    unfreeze_last_block: bool = False  # mở layer4 ResNet (thí nghiệm phụ)
    num_heads: int = 8          # số head cho cross_attention
    hidden_dim: int = 1024      # chiều ẩn của MLP head
    dropout: float = 0.5

    # ----- Training -----
    batch_size: int = 64
    lr: float = 1e-3            # cho phần tự xây: fusion + projection + head
    lr_backbone: float = 1e-5   # cho layer4 ResNet khi unfreeze_last_block
    weight_decay: float = 1e-2
    max_epochs: int = 30
    patience: int = 5           # early stopping theo val overall accuracy
    seed: int = 42
    num_workers: int = 2

    # ----- IO -----
    run_name: str = ""          # rỗng → tự đặt = tên fusion
    output_dir: str = str(MIDTERM_DIR / "outputs")
    checkpoint_dir: str = str(MIDTERM_DIR / "checkpoints")

    def __post_init__(self):
        if not self.run_name:
            self.run_name = self.fusion
        assert self.fusion in ("concat", "hadamard", "cross_attention")
        assert self.text_pool in ("mean", "cls")

    def to_dict(self) -> dict:
        return asdict(self)


def pick_device():
    """Chọn device theo thứ tự ưu tiên: cuda (Colab GPU) → mps (Mac) → cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Verify**

Run:
```bash
python -c "
from midterm.config import Config, pick_device
cfg = Config(fusion='cross_attention')
print(cfg.run_name, cfg.text_pool, cfg.d_model)
print(pick_device())
"
```
Expected: `cross_attention mean 768` và device `mps` (local Mac).

- [ ] **Step 5: Commit**

```bash
git add midterm/requirements.txt midterm/__init__.py midterm/data/__init__.py midterm/models/__init__.py midterm/config.py
git commit -m "feat(midterm): scaffold package + central Config dataclass"
```

---

### Task 2: data/download.py — tải VQA-RAD idempotent

**Files:**
- Create: `midterm/data/download.py`

- [ ] **Step 1: Viết `midterm/data/download.py`**

```python
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
```

- [ ] **Step 2: Verify (nhánh idempotent — dataset đã có sẵn trên đĩa)**

Run: `python -m midterm.data.download`
Expected: `Dataset đã có tại .../midterm/data/vqa_rad — bỏ qua download.` (không tải lại)

- [ ] **Step 3: Commit**

```bash
git add midterm/data/download.py
git commit -m "feat(midterm): idempotent VQA-RAD download script"
```

---

### Task 3: data/vocab.py — answer vocab 429 class, không `<unk>`

**Files:**
- Create: `midterm/data/vocab.py`
- Overwrite (artifact, gitignored): `midterm/data/answer_vocab.json`

- [ ] **Step 1: Viết `midterm/data/vocab.py`**

```python
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
```

- [ ] **Step 2: Verify — build lại vocab, ghi đè file cũ (file cũ có `<unk>`, 433 entries)**

Run: `python -m midterm.data.vocab`
Expected:
```
Vocab: 429 class → .../midterm/data/answer_vocab.json
Độ phủ vocab trên test: 334/451 = 74.1%
```

Run thêm: `python -c "from midterm.data.vocab import load_vocab; from midterm.config import Config; v = load_vocab(Config().vocab_path); print(len(v), '<unk>' in v)"`
Expected: `429 False`

- [ ] **Step 3: Commit**

```bash
git add midterm/data/vocab.py
git commit -m "feat(midterm): answer vocab builder (429 classes, no unk, prints test coverage)"
```

---

### Task 4: data/dataset.py — Dataset + transforms + val split theo QA pair

**Files:**
- Create: `midterm/data/dataset.py`

- [ ] **Step 1: Viết `midterm/data/dataset.py`**

```python
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
```

- [ ] **Step 2: Verify shapes và split sizes**

Run:
```bash
python -c "
from transformers import AutoTokenizer
from midterm.config import Config
from midterm.data.dataset import VQARadDataset, load_splits
from midterm.data.vocab import load_vocab

cfg = Config()
vocab = load_vocab(cfg.vocab_path)
tok = AutoTokenizer.from_pretrained(cfg.text_model_name)
train_hf, val_hf, test_hf = load_splits(cfg)
print('splits:', len(train_hf), len(val_hf), len(test_hf))
sample = VQARadDataset(train_hf, tok, vocab, cfg, train=True)[0]
print('image:', tuple(sample['image'].shape))
print('input_ids:', tuple(sample['input_ids'].shape))
print('label trong vocab:', sample['label'].item() >= 0)
"
```
Expected:
```
splits: 1613 180 451
image: (3, 224, 224)
input_ids: (32,)
label trong vocab: True
```

- [ ] **Step 3: Commit**

```bash
git add midterm/data/dataset.py
git commit -m "feat(midterm): VQARadDataset + transforms + QA-pair val split"
```

---

### Task 5: models/image_encoder.py — ResNet-50 frozen, BN luôn eval

**Files:**
- Create: `midterm/models/image_encoder.py`

- [ ] **Step 1: Viết `midterm/models/image_encoder.py`**

```python
"""Image encoder: ResNet-50 pretrained ImageNet, mặc định freeze toàn bộ.

Xuất 2 dạng đặc trưng:
- v_img   (B, d_model): vector toàn cục (avg pool) — cho fusion concat/hadamard.
- img_map (B, 49, d_model): 7×7 = 49 vùng không gian — cho cross_attention.
Cả hai chiếu qua Linear riêng về d_model để khớp chiều với vector câu hỏi.
"""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ImageEncoder(nn.Module):
    def __init__(self, d_model: int = 768, unfreeze_last_block: bool = False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Bỏ avgpool + fc cuối, giữ phần conv → feature map (B, 2048, 7, 7)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            # layer4 (block conv sâu nhất) là child cuối của backbone;
            # train với LR riêng thấp hơn — xem param groups trong train.py
            for p in self.backbone[-1].parameters():
                p.requires_grad = True

        self.proj_global = nn.Linear(2048, d_model)
        self.proj_regions = nn.Linear(2048, d_model)

    def train(self, mode: bool = True):
        """BatchNorm của backbone LUÔN ở eval mode, kể cả lúc train.

        requires_grad=False chỉ chặn cập nhật weight qua optimizer; nếu để
        train mode, BatchNorm vẫn cập nhật running mean/var theo ảnh y khoa →
        encoder "frozen" âm thầm thay đổi hành vi và kết quả không tái lập.
        Gradient vẫn chảy bình thường qua module ở eval mode (cần cho
        unfreeze_last_block).
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor):
        fmap = self.backbone(images)               # (B, 2048, 7, 7)
        v_global = fmap.mean(dim=(2, 3))           # avg pool → (B, 2048)
        regions = fmap.flatten(2).transpose(1, 2)  # (B, 49, 2048)
        return self.proj_global(v_global), self.proj_regions(regions)
```

- [ ] **Step 2: Verify shapes + BN giữ eval mode sau `.train()`**

Run:
```bash
python -c "
import torch
from midterm.models.image_encoder import ImageEncoder

enc = ImageEncoder()
enc.train()  # giả lập training loop gọi model.train()
v_img, img_map = enc(torch.randn(2, 3, 224, 224))
print('v_img:', tuple(v_img.shape), '| img_map:', tuple(img_map.shape))
bn = [m for m in enc.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
print('BN modules:', len(bn), '| tất cả eval:', all(not m.training for m in bn))
frozen = all(not p.requires_grad for p in enc.backbone.parameters())
print('backbone frozen:', frozen)
"
```
Expected:
```
v_img: (2, 768) | img_map: (2, 49, 768)
BN modules: 53 | tất cả eval: True
backbone frozen: True
```

- [ ] **Step 3: Commit**

```bash
git add midterm/models/image_encoder.py
git commit -m "feat(midterm): frozen ResNet-50 image encoder, BN pinned to eval"
```

---

### Task 6: models/text_encoder.py — BERT frozen, mean/cls pooling

**Files:**
- Create: `midterm/models/text_encoder.py`

- [ ] **Step 1: Viết `midterm/models/text_encoder.py`**

```python
"""Text encoder: BERT pretrained, freeze toàn bộ.

Vector câu hỏi 768-d lấy theo config text_pool:
- "mean" (mặc định): mean-pooling hidden state lớp cuối theo attention mask.
  [CLS] của BERT freeze vốn được pretrain cho next-sentence prediction nên là
  biểu diễn câu yếu — mean-pooling thường tốt hơn rõ rệt khi không fine-tune.
- "cls": embedding [CLS] — giữ làm đối chứng (thí nghiệm phụ).
"""
import torch
from torch import nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pool: str = "mean"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pool = pool
        for p in self.bert.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # Freeze hoàn toàn → luôn eval: tắt dropout bên trong BERT để biểu diễn
        # câu hỏi ổn định giữa các batch (cùng lý do BN ở image encoder).
        super().train(mode)
        self.bert.eval()
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                   # (B, L, 768)
        if self.pool == "cls":
            return hidden[:, 0]
        # mean-pooling: chỉ trung bình trên token thật (mask=1), bỏ padding
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```

- [ ] **Step 2: Verify shape, mean ≠ cls, frozen**

Run:
```bash
python -c "
import torch
from transformers import AutoTokenizer
from midterm.models.text_encoder import TextEncoder

tok = AutoTokenizer.from_pretrained('bert-base-uncased')
batch = tok(['is there cardiomegaly?', 'what plane is this?'],
            padding='max_length', truncation=True, max_length=32, return_tensors='pt')
mean_enc, cls_enc = TextEncoder(pool='mean'), TextEncoder(pool='cls')
v_mean = mean_enc(batch['input_ids'], batch['attention_mask'])
v_cls = cls_enc(batch['input_ids'], batch['attention_mask'])
print('shape:', tuple(v_mean.shape))
print('mean khac cls:', not torch.allclose(v_mean, v_cls))
print('frozen:', all(not p.requires_grad for p in mean_enc.parameters()))
"
```
Expected:
```
shape: (2, 768)
mean khac cls: True
frozen: True
```

- [ ] **Step 3: Commit**

```bash
git add midterm/models/text_encoder.py
git commit -m "feat(midterm): frozen BERT text encoder with mean/cls pooling"
```

---

### Task 7: models/fusion.py — 3 chiến lược fusion cùng interface

**Files:**
- Create: `midterm/models/fusion.py`

- [ ] **Step 1: Viết `midterm/models/fusion.py`**

```python
"""Ba chiến lược fusion — trục ablation chính của đồ án.

Interface chung: forward(v_img, img_map, v_txt) → (fused, attn_weights | None)
- v_img   (B, d): vector ảnh toàn cục
- img_map (B, 49, d): 49 vùng không gian của ảnh (chỉ cross_attention dùng)
- v_txt   (B, d): vector câu hỏi
- fused   (B, d): vector hợp nhất đưa vào MLP head
- attn_weights (B, 49): chỉ có ở cross_attention — để vẽ heatmap minh họa.

Lưu ý diễn giải ablation: cross_attention nhận thêm thông tin spatial (img_map)
mà concat/hadamard không dùng — khác biệt kết quả gộp cả "cơ chế fusion" lẫn
"có thông tin spatial" (xem spec mục 3.3). Báo cáo phải nêu confound này.
"""
import torch
from torch import nn


class ConcatFusion(nn.Module):
    """Baseline đơn giản nhất: nối [v_img ; v_txt] rồi chiếu tuyến tính + ReLU."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.fc = nn.Linear(2 * d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, v_img, img_map, v_txt):
        return self.relu(self.fc(torch.cat([v_img, v_txt], dim=-1))), None


class HadamardFusion(nn.Module):
    """Tương tác nhân: chiếu mỗi modality qua Linear riêng rồi nhân từng phần tử.
    Phép nhân buộc hai modality "đồng thuận" theo từng chiều — feature chỉ lớn
    khi cả ảnh lẫn câu hỏi cùng kích hoạt chiều đó."""

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.proj_img = nn.Linear(d_model, d_model)
        self.proj_txt = nn.Linear(d_model, d_model)

    def forward(self, v_img, img_map, v_txt):
        return self.proj_img(v_img) * self.proj_txt(v_txt), None


class CrossAttentionFusion(nn.Module):
    """Câu hỏi (query duy nhất) "nhìn" vào 49 vùng ảnh (key/value) bằng
    multi-head attention, cộng residual với v_txt rồi LayerNorm — một block
    Transformer tối giản. Attention weights cho biết model nhìn vào đâu."""

    def __init__(self, d_model: int = 768, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, v_img, img_map, v_txt):
        query = v_txt.unsqueeze(1)                              # (B, 1, d)
        attended, weights = self.attn(query, img_map, img_map)  # (B,1,d), (B,1,49)
        fused = self.norm(v_txt + attended.squeeze(1))          # residual + LN
        return fused, weights.squeeze(1)                        # attn: (B, 49)


def build_fusion(cfg):
    """Factory: đổi fusion = đổi config, không đụng phần code khác."""
    if cfg.fusion == "concat":
        return ConcatFusion(cfg.d_model)
    if cfg.fusion == "hadamard":
        return HadamardFusion(cfg.d_model)
    return CrossAttentionFusion(cfg.d_model, cfg.num_heads)
```

- [ ] **Step 2: Verify cả 3 fusion cùng interface**

Run:
```bash
python -c "
import torch
from midterm.config import Config
from midterm.models.fusion import build_fusion

v_img, img_map, v_txt = torch.randn(2, 768), torch.randn(2, 49, 768), torch.randn(2, 768)
for name in ('concat', 'hadamard', 'cross_attention'):
    fusion = build_fusion(Config(fusion=name))
    fused, attn = fusion(v_img, img_map, v_txt)
    print(f'{name}: fused {tuple(fused.shape)}, attn', None if attn is None else tuple(attn.shape))
"
```
Expected:
```
concat: fused (2, 768), attn None
hadamard: fused (2, 768), attn None
cross_attention: fused (2, 768), attn (2, 49)
```

- [ ] **Step 3: Commit**

```bash
git add midterm/models/fusion.py
git commit -m "feat(midterm): three fusion strategies behind one interface"
```

---

### Task 8: models/vqa_model.py — ghép model hoàn chỉnh

**Files:**
- Create: `midterm/models/vqa_model.py`

- [ ] **Step 1: Viết `midterm/models/vqa_model.py`**

```python
"""VQAModel: ghép 4 thành phần theo config — image encoder + text encoder
+ fusion + MLP head. Encoder freeze nên phần trainable chỉ vài triệu tham số,
train được trên T4 trong vài phút mỗi epoch.
"""
import torch
from torch import nn

from midterm.models.fusion import build_fusion
from midterm.models.image_encoder import ImageEncoder
from midterm.models.text_encoder import TextEncoder


class VQAModel(nn.Module):
    def __init__(self, cfg, num_classes: int):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg.d_model, cfg.unfreeze_last_block)
        self.text_encoder = TextEncoder(cfg.text_model_name, cfg.text_pool)
        self.fusion = build_fusion(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),  # 0.5 — chống overfit trên 1.6k mẫu train
            nn.Linear(cfg.hidden_dim, num_classes),
        )

    def forward(self, images, input_ids, attention_mask):
        v_img, img_map = self.image_encoder(images)
        v_txt = self.text_encoder(input_ids, attention_mask)
        fused, attn = self.fusion(v_img, img_map, v_txt)
        return self.head(fused), attn

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return total, trainable
```

- [ ] **Step 2: Verify forward end-to-end + số tham số trainable**

Run:
```bash
python -c "
import torch
from transformers import AutoTokenizer
from midterm.config import Config
from midterm.models.vqa_model import VQAModel

tok = AutoTokenizer.from_pretrained('bert-base-uncased')
batch = tok(['is there cardiomegaly?'] * 2, padding='max_length', truncation=True,
            max_length=32, return_tensors='pt')
for name in ('concat', 'cross_attention'):
    model = VQAModel(Config(fusion=name), num_classes=429)
    logits, attn = model(torch.randn(2, 3, 224, 224),
                         batch['input_ids'], batch['attention_mask'])
    total, trainable = model.count_parameters()
    print(f'{name}: logits {tuple(logits.shape)}, trainable {trainable/1e6:.1f}M / total {total/1e6:.0f}M')
"
```
Expected: logits `(2, 429)` cho cả hai; trainable trong khoảng **4–8M**; total ~**135M** (ResNet 25M + BERT 110M).

- [ ] **Step 3: Commit**

```bash
git add midterm/models/vqa_model.py
git commit -m "feat(midterm): assemble VQAModel from config"
```

---

### Task 9: train.py — training loop + smoke test MPS

**Files:**
- Create: `midterm/train.py`

- [ ] **Step 1: Viết `midterm/train.py`**

```python
"""Training loop cho MedVQA.

Ba thí nghiệm chính của báo cáo (chỉ khác fusion, cùng seed):

    python -m midterm.train --fusion concat
    python -m midterm.train --fusion hadamard
    python -m midterm.train --fusion cross_attention

Smoke test local (subset 128 mẫu, 2 epoch, run_name có hậu tố _smoke để không
ghi đè checkpoint thật):

    python -m midterm.train --fusion concat --smoke

Mỗi run lưu: outputs/<run_name>/{config.json, history.json, curves.png}
và checkpoint tốt nhất (theo val accuracy) tại checkpoints/<run_name>.pt.
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render PNG không cần GUI (chạy được trên server/Colab)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from midterm.config import Config, pick_device
from midterm.data.dataset import VQARadDataset, load_splits
from midterm.data.vocab import load_vocab
from midterm.models.vqa_model import VQAModel


def set_seed(seed: int) -> None:
    """Fix mọi nguồn ngẫu nhiên: 3 thí nghiệm fusion chỉ khác nhau ở fusion,
    không khác ở khởi tạo hay thứ tự shuffle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_validation(model, loader, device):
    """Trả về (accuracy, loss trung bình) trên loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(images, input_ids, attention_mask)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    return correct / total, loss_sum / total


def plot_curves(history: dict, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_xlabel("epoch"), ax1.set_title("Loss"), ax1.legend()
    ax2.plot(epochs, history["val_acc"])
    ax2.set_xlabel("epoch"), ax2.set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: Config, smoke: bool = False) -> float:
    set_seed(cfg.seed)
    device = pick_device()
    print(f"Device: {device} | run: {cfg.run_name} | fusion: {cfg.fusion}"
          f" | text_pool: {cfg.text_pool}")

    vocab = load_vocab(cfg.vocab_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    train_hf, val_hf, _ = load_splits(cfg)
    train_ds = VQARadDataset(train_hf, tokenizer, vocab, cfg, train=True)
    val_ds = VQARadDataset(val_hf, tokenizer, vocab, cfg, train=False)
    if smoke:  # subset nhỏ: chỉ kiểm tra pipeline chạy end-to-end
        train_ds = Subset(train_ds, range(128))
        val_ds = Subset(val_ds, range(32))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = VQAModel(cfg, num_classes=len(vocab)).to(device)
    total, trainable = model.count_parameters()
    print(f"Tham số: {total / 1e6:.0f}M tổng | {trainable / 1e6:.1f}M trainable")

    # Hai param group: phần tự xây (fusion + projection + head) học LR cao;
    # layer4 của ResNet (nếu unfreeze) học LR thấp hơn 100 lần để không phá
    # feature pretrained bằng gradient lớn lúc đầu.
    backbone_params = [p for p in model.image_encoder.backbone.parameters()
                       if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    new_params = [p for p in model.trainable_parameters()
                  if id(p) not in backbone_ids]
    param_groups = [{"params": new_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, epochs_no_improve = 0.0, 0
    max_epochs = 2 if smoke else cfg.max_epochs

    for epoch in range(1, max_epochs + 1):
        model.train()  # encoder tự ghim eval bên trong (xem image/text encoder)
        loss_sum, seen = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        val_acc, val_loss = run_validation(model, val_loader, device)
        history["train_loss"].append(loss_sum / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d} | train loss {loss_sum / seen:.4f}"
              f" | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

        # Early stopping theo val overall accuracy; lưu checkpoint tốt nhất
        if val_acc > best_acc:
            best_acc, epochs_no_improve = val_acc, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
                "num_classes": len(vocab),
                "epoch": epoch,
                "val_acc": val_acc,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping tại epoch {epoch} (patience {cfg.patience})")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, out_dir / "curves.png")
    print(f"Best val acc: {best_acc:.4f} | checkpoint: {ckpt_path}")
    return best_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MedVQA trên VQA-RAD")
    parser.add_argument("--fusion", default="concat",
                        choices=["concat", "hadamard", "cross_attention"])
    parser.add_argument("--run-name", default="")
    parser.add_argument("--text-pool", default="mean", choices=["mean", "cls"])
    parser.add_argument("--unfreeze-last-block", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="subset 128 mẫu + 2 epoch: kiểm tra pipeline end-to-end")
    args = parser.parse_args()

    run_name = args.run_name or (f"{args.fusion}_smoke" if args.smoke else args.fusion)
    cfg = Config(fusion=args.fusion, run_name=run_name, text_pool=args.text_pool,
                 unfreeze_last_block=args.unfreeze_last_block)
    if args.smoke:
        cfg.batch_size = 16
        cfg.num_workers = 0
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify — smoke run trên MPS**

Run: `python -m midterm.train --fusion concat --smoke`
Expected: in device `mps`, tham số `~135M tổng | ~5.x M trainable`, 2 dòng epoch với loss giảm, kết thúc với `Best val acc: ...` và checkpoint `midterm/checkpoints/concat_smoke.pt`. Thời gian ~1–3 phút.

Run tiếp:
```bash
ls midterm/outputs/concat_smoke/ && ls midterm/checkpoints/
```
Expected: `config.json curves.png history.json` và `concat_smoke.pt`.

- [ ] **Step 3: Commit**

```bash
git add midterm/train.py
git commit -m "feat(midterm): training loop with early stopping, param groups, smoke mode"
```

---

### Task 10: evaluate.py — overall/closed/open + bảng ví dụ

**Files:**
- Create: `midterm/evaluate.py`

- [ ] **Step 1: Viết `midterm/evaluate.py`**

```python
"""Đánh giá checkpoint trên test split (451 mẫu) — chỉ chạy ở bước cuối.

Ba số liệu chuẩn của VQA-RAD:
- overall: accuracy trên toàn bộ test
- closed:  accuracy trên câu hỏi yes/no (đáp án chuẩn hóa thuộc {yes, no})
- open:    accuracy trên phần còn lại

Chạy: python -m midterm.evaluate --checkpoint midterm/checkpoints/concat.pt
Kết quả chi tiết lưu vào outputs/<run_name>/test_results.json — nguyên liệu
cho phần phân tích lỗi của báo cáo.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from midterm.config import Config, pick_device
from midterm.data.dataset import VQARadDataset, load_splits
from midterm.data.vocab import load_vocab
from midterm.models.vqa_model import VQAModel


def load_model(checkpoint_path: str, device):
    """Dựng lại model từ config lưu trong checkpoint — evaluate/demo không cần
    biết run đó dùng fusion hay text_pool nào."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = VQAModel(cfg, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def evaluate(checkpoint: str) -> dict:
    device = pick_device()
    model, cfg = load_model(checkpoint, device)
    vocab = load_vocab(cfg.vocab_path)
    idx_to_answer = {idx: ans for ans, idx in vocab.items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    _, _, test_hf = load_splits(cfg)
    test_ds = VQARadDataset(test_hf, tokenizer, vocab, cfg, train=False)
    loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=0)

    rows = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(batch["image"].to(device),
                              batch["input_ids"].to(device),
                              batch["attention_mask"].to(device))
            preds = logits.argmax(dim=-1).cpu()
            for question, answer, label, pred in zip(
                    batch["question"], batch["answer"], batch["label"], preds):
                rows.append({
                    "question": question,
                    "answer": answer,
                    "pred": idx_to_answer[pred.item()],
                    # label -1 (đáp án ngoài vocab) không bao giờ khớp → tính sai
                    "correct": pred.item() == label.item(),
                    "closed": answer in ("yes", "no"),
                })

    def accuracy(subset):
        return sum(r["correct"] for r in subset) / len(subset) if subset else 0.0

    closed_rows = [r for r in rows if r["closed"]]
    open_rows = [r for r in rows if not r["closed"]]
    in_vocab = sum(r["answer"] in vocab for r in rows)
    metrics = {
        "overall": accuracy(rows),
        "closed": accuracy(closed_rows),
        "open": accuracy(open_rows),
        "n_test": len(rows),
        "n_closed": len(closed_rows),
        "n_open": len(open_rows),
        "vocab_coverage": in_vocab / len(rows),
    }

    print(f"Checkpoint: {checkpoint} (run: {cfg.run_name}, fusion: {cfg.fusion})")
    print(f"Overall: {metrics['overall']:.4f} (n={metrics['n_test']})")
    print(f"Closed (yes/no): {metrics['closed']:.4f} (n={metrics['n_closed']})")
    print(f"Open: {metrics['open']:.4f} (n={metrics['n_open']})")
    print(f"Độ phủ vocab trên test: {metrics['vocab_coverage']:.1%}"
          f" — trần accuracy khả dĩ")

    # Bảng ví dụ đúng/sai — nguyên liệu phân tích lỗi cho báo cáo
    def show(title, subset):
        print(f"\n--- {title} ---")
        for r in subset[:5]:
            print(f"  Q: {r['question'][:60]:60s} | gt: {r['answer'][:20]:20s}"
                  f" | pred: {r['pred'][:20]}")

    show("Ví dụ dự đoán ĐÚNG", [r for r in rows if r["correct"]])
    show("Ví dụ dự đoán SAI", [r for r in rows if not r["correct"]])

    out_path = Path(cfg.output_dir) / cfg.run_name / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "rows": rows},
                                   indent=2, ensure_ascii=False))
    print(f"\nKết quả chi tiết → {out_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MedVQA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
```

- [ ] **Step 2: Verify trên smoke checkpoint từ Task 9**

Run: `python -m midterm.evaluate --checkpoint midterm/checkpoints/concat_smoke.pt`
Expected: in đủ 4 dòng metrics (accuracy thấp là bình thường — model smoke chỉ train 128 mẫu × 2 epoch), `n=451`, `n_closed + n_open = 451`, độ phủ `74.1%`, hai bảng ví dụ, và file `midterm/outputs/concat_smoke/test_results.json` được tạo.

- [ ] **Step 3: Commit**

```bash
git add midterm/evaluate.py
git commit -m "feat(midterm): test evaluation with overall/closed/open metrics"
```

---

### Task 11: demo.py — inference + attention overlay

**Files:**
- Create: `midterm/demo.py`

- [ ] **Step 1: Viết `midterm/demo.py`**

```python
"""Demo inference: một ảnh + một câu hỏi → top-5 đáp án kèm xác suất.

Chạy: python -m midterm.demo --checkpoint midterm/checkpoints/cross_attention.pt \
          --image chest.jpg --question "is there cardiomegaly?"

Với checkpoint cross_attention: lưu thêm attention overlay (heatmap 7×7 phóng
to chồng lên ảnh) — minh họa model "nhìn" vào vùng nào khi trả lời, dùng cho
báo cáo và vấn đáp.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from midterm.config import pick_device
from midterm.data.dataset import build_transforms
from midterm.data.vocab import load_vocab
from midterm.evaluate import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="MedVQA demo inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", default="demo_attention.png",
                        help="đường dẫn lưu attention overlay (chỉ cross_attention)")
    args = parser.parse_args()

    device = pick_device()
    model, cfg = load_model(args.checkpoint, device)
    vocab = load_vocab(cfg.vocab_path)
    idx_to_answer = {idx: ans for ans, idx in vocab.items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)

    image = Image.open(args.image).convert("RGB")
    pixel = build_transforms(cfg, train=False)(image).unsqueeze(0).to(device)
    tokens = tokenizer(args.question, padding="max_length", truncation=True,
                       max_length=cfg.max_question_len, return_tensors="pt")

    with torch.no_grad():
        logits, attn = model(pixel,
                             tokens["input_ids"].to(device),
                             tokens["attention_mask"].to(device))

    probs = logits.softmax(dim=-1).squeeze(0)
    top = probs.topk(5)
    print(f"Q: {args.question}")
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {idx_to_answer[idx]:<30s} {prob:.3f}")

    if attn is not None:
        # attn (1, 49) → lưới 7×7 → phóng to 32× bằng np.kron → 224×224
        heat = attn.squeeze(0).reshape(7, 7).cpu().numpy()
        heat_big = np.kron(heat, np.ones((32, 32)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image.resize((224, 224)))
        ax.imshow(heat_big, cmap="jet", alpha=0.4)
        ax.axis("off")
        ax.set_title(args.question, fontsize=9)
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Attention overlay → {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-train cross_attention để có checkpoint cho overlay**

Run: `python -m midterm.train --fusion cross_attention --smoke`
Expected: hoàn thành 2 epoch, tạo `midterm/checkpoints/cross_attention_smoke.pt`.

- [ ] **Step 3: Verify demo — trích 1 ảnh test làm input**

Run:
```bash
python -c "
from datasets import load_from_disk
from midterm.config import Config
ds = load_from_disk(Config().data_dir)
ds['test'][0]['image'].convert('RGB').save('/tmp/vqa_sample.jpg')
print('Q mẫu:', ds['test'][0]['question'])
"
python -m midterm.demo --checkpoint midterm/checkpoints/cross_attention_smoke.pt \
    --image /tmp/vqa_sample.jpg --question "is there evidence of an aortic aneurysm?" \
    --out /tmp/demo_attention.png
ls -la /tmp/demo_attention.png
```
Expected: in top-5 đáp án kèm xác suất (tổng ≤ 1, đáp án nằm trong vocab), dòng `Attention overlay → /tmp/demo_attention.png`, file PNG tồn tại.

- [ ] **Step 4: Commit**

```bash
git add midterm/demo.py
git commit -m "feat(midterm): demo CLI with top-5 answers and attention overlay"
```

---

### Task 12: colab_train.ipynb — notebook mỏng cho Colab

**Files:**
- Create: `midterm/colab_train.ipynb`

- [ ] **Step 1: Viết notebook** — chỉ gọi CLI, không viết lại logic. Tạo bằng Write tool với JSON ipynb (nbformat 4). Nội dung các cell theo đúng thứ tự:

Cell 1 (markdown):
```markdown
# MedVQA trên VQA-RAD — train trên Colab GPU

Notebook mỏng: mọi logic nằm trong `midterm/` modules, ở đây chỉ gọi CLI.
Bật GPU: Runtime → Change runtime type → T4 GPU.
```

Cell 2 (code — clone + cài đặt):
```python
!git clone https://github.com/tuan1kdt/deep-learning.git
%cd deep-learning
!pip install -q -r midterm/requirements.txt
```

Cell 3 (code — chuẩn bị data):
```python
!python -m midterm.data.download
!python -m midterm.data.vocab
```

Cell 4 (code — train 3 thí nghiệm fusion):
```python
!python -m midterm.train --fusion concat
!python -m midterm.train --fusion hadamard
!python -m midterm.train --fusion cross_attention
```

Cell 5 (code — đánh giá trên test):
```python
!python -m midterm.evaluate --checkpoint midterm/checkpoints/concat.pt
!python -m midterm.evaluate --checkpoint midterm/checkpoints/hadamard.pt
!python -m midterm.evaluate --checkpoint midterm/checkpoints/cross_attention.pt
```

Cell 6 (code — demo + attention overlay):
```python
from datasets import load_from_disk
ds = load_from_disk("midterm/data/vqa_rad")
sample = ds["test"][0]
sample["image"].convert("RGB").save("/tmp/vqa_sample.jpg")
print("Q:", sample["question"], "| GT:", sample["answer"])

!python -m midterm.demo --checkpoint midterm/checkpoints/cross_attention.pt \
    --image /tmp/vqa_sample.jpg --question "{sample['question']}" \
    --out demo_attention.png

from IPython.display import Image as IPImage
IPImage("demo_attention.png")
```

Cell 7 (code — tải artifacts về máy):
```python
!zip -r results.zip midterm/outputs midterm/checkpoints
from google.colab import files
files.download("results.zip")
```

JSON khung notebook (mỗi cell code có `"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": null`, source là list dòng có `\n` cuối):
```json
{
  "nbformat": 4,
  "nbformat_minor": 2,
  "metadata": {
    "accelerator": "GPU",
    "language_info": {"name": "python"}
  },
  "cells": [ ... 7 cell như trên ... ]
}
```

- [ ] **Step 2: Verify JSON hợp lệ và đủ cell**

Run:
```bash
python -c "
import json
nb = json.load(open('midterm/colab_train.ipynb'))
print('cells:', len(nb['cells']), '| types:', [c['cell_type'] for c in nb['cells']])
"
```
Expected: `cells: 7 | types: ['markdown', 'code', 'code', 'code', 'code', 'code', 'code']`

- [ ] **Step 3: Commit**

```bash
git add midterm/colab_train.ipynb
git commit -m "feat(midterm): thin Colab notebook driving the CLI pipeline"
```

---

### Task 13: README.md + smoke hadamard + dọn artifacts smoke

**Files:**
- Modify: `midterm/README.md` (đọc nội dung hiện có trước, viết lại theo spec mục 6)

- [ ] **Step 1: Smoke-train fusion còn lại để xác nhận cả 3 đường chạy**

Run: `python -m midterm.train --fusion hadamard --smoke`
Expected: hoàn thành 2 epoch không lỗi.

- [ ] **Step 2: Viết lại `midterm/README.md`** (tiếng Việt, theo phong cách `dayN/README.md`) với các mục:
  - Bài toán + dataset (VQA-RAD, 1.793/451, 313 ảnh duy nhất, 429 class, coverage 74,1%).
  - Sơ đồ kiến trúc (ASCII từ spec mục 3) + giải thích ngắn từng thành phần và link file.
  - Cách chạy: download → vocab → train ×3 → evaluate → demo (lệnh đầy đủ), smoke test, notebook Colab.
  - Bảng kết quả 3 fusion (overall/closed/open) — **để trống có ghi chú "điền sau khi train trên Colab"**.
  - Ghi chú thiết kế cho vấn đáp: vì sao BN ghim eval, vì sao mean-pooling, vì sao val split theo QA pair không phải leakage, confound của ablation cross_attention.

- [ ] **Step 3: Xóa artifacts smoke (đều gitignored, chỉ để sạch thư mục)**

```bash
rm -rf midterm/outputs/concat_smoke midterm/outputs/hadamard_smoke midterm/outputs/cross_attention_smoke
rm -f midterm/checkpoints/concat_smoke.pt midterm/checkpoints/hadamard_smoke.pt midterm/checkpoints/cross_attention_smoke.pt
```

- [ ] **Step 4: Verify tổng — git status sạch artifacts, chỉ còn file nguồn**

Run: `git status --short midterm/`
Expected: chỉ thấy các file `.py`, `.ipynb`, `README.md`, `requirements.txt` (đã commit hết → output rỗng); không thấy `outputs/`, `checkpoints/`, `*.json` data.

- [ ] **Step 5: Commit**

```bash
git add midterm/README.md
git commit -m "docs(midterm): README with architecture, usage, and design rationale"
```

---

## Self-Review (đã chạy)

1. **Spec coverage:** §2 ràng buộc → Task 1 (config) + 12 (Colab); §3.1 → Task 5; §3.2 → Task 6; §3.3 → Task 7; §3.4 → Task 8; §4.1 → Task 2; §4.2 → Task 3; §4.3 → Task 4; §5.1 → Task 9; §5.2 → Task 10; §5.3 → Task 11; §6 cấu trúc + notebook → Task 12–13; §7 rủi ro: overfit (dropout/early stop/augment — Task 4, 8, 9), imbalance (closed/open — Task 10), OOV (coverage — Task 3, 10). Không có gap.
2. **Placeholder scan:** bảng kết quả README để trống là chủ đích (số liệu chỉ có sau khi train thật trên Colab) và được ghi chú rõ — không phải placeholder kế hoạch.
3. **Type consistency:** interface fusion `(v_img, img_map, v_txt) → (fused, attn|None)` dùng nhất quán ở Task 7, 8, 10, 11; checkpoint dict (`model_state`, `config`, `num_classes`) khớp giữa Task 9 (save) và Task 10 (load — `torch.load` mặc định `weights_only=True` của torch ≥2.6 chấp nhận dict toàn tensor/primitive này); `run_name` smoke có hậu tố `_smoke` nhất quán Task 9, 10, 11, 13.
