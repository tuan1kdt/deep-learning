"""Cấu hình trung tâm cho đồ án cuối kỳ (image captioning Flickr8k).

Theo khuôn midterm: mọi lựa chọn thí nghiệm đi qua dataclass Config —
đổi thí nghiệm = đổi CLI flag, không sửa code model hay training loop.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

# Neo mọi đường dẫn theo thư mục final/ để chạy được từ bất kỳ cwd nào
# (local lẫn Colab) — cùng lý do với MIDTERM_DIR bên midterm/.
FINAL_DIR = Path(__file__).resolve().parent

# LR/warmup mặc định khác nhau theo họ decoder: Transformer trên data nhỏ
# cần warmup để không phá pre-norm lúc đầu; LSTM thì 3e-4 ổn định sẵn.
_AUTO_LR = {"lstm": 3e-4, "transformer": 1e-4}
_AUTO_WARMUP = {"lstm": 0, "transformer": 2000}


@dataclass
class Config:
    # ----- Data -----
    hf_dataset: str = "jxie/flickr8k"
    dataset_dir: str = str(FINAL_DIR / "data" / "flickr8k")
    data_root: str = str(FINAL_DIR / "data")
    vocab_path: str = str(FINAL_DIR / "data" / "vocab.json")
    min_word_freq: int = 5
    max_words: int = 20         # số từ tối đa của caption (chưa tính bos/eos)

    # ----- Model -----
    decoder: str = "lstm"       # lstm | transformer
    use_attention: bool = True  # chỉ có nghĩa với lstm (ablation thí nghiệm #2)
    d_model: int = 512
    attn_dim: int = 512         # chiều ẩn của Bahdanau attention
    num_layers: int = 3         # số block Transformer
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.3
    feat_dim: int = 2048        # ResNet-50 layer4
    num_regions: int = 49       # 7x7 vùng không gian

    # ----- Training -----
    batch_size: int = 128
    lr: float = 0.0             # 0 = auto theo decoder (xem _AUTO_LR)
    warmup_steps: int = -1      # -1 = auto theo decoder
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    max_epochs: int = 25
    patience: int = 5           # early stopping theo val loss
    seed: int = 42
    num_workers: int = 2

    # ----- Eval -----
    beam_sizes: tuple = (3, 5)

    # ----- IO -----
    run_name: str = ""          # rỗng → tự đặt theo decoder (+_noattn)
    output_dir: str = str(FINAL_DIR / "outputs")
    checkpoint_dir: str = str(FINAL_DIR / "checkpoints")

    def __post_init__(self):
        assert self.decoder in ("lstm", "transformer")
        if self.lr == 0.0:
            self.lr = _AUTO_LR[self.decoder]
        if self.warmup_steps < 0:
            self.warmup_steps = _AUTO_WARMUP[self.decoder]
        if not self.run_name:
            suffix = "" if self.use_attention else "_noattn"
            self.run_name = f"{self.decoder}{suffix}"

    @property
    def max_len(self) -> int:
        """Độ dài câu đầy đủ tối đa: bos + max_words + eos."""
        return self.max_words + 2

    def features_path(self, split: str) -> Path:
        return Path(self.data_root) / f"features_{split}.pt"

    def captions_path(self, split: str) -> Path:
        return Path(self.data_root) / f"captions_{split}.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["beam_sizes"] = list(d["beam_sizes"])  # tuple không JSON-serializable
        return d


def pick_device():
    """cuda (Colab GPU) → mps (Mac) → cpu, đúng thứ tự ưu tiên như midterm."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
