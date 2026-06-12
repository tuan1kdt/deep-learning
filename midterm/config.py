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
