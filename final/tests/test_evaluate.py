import torch

from final.config import Config
from final.data.vocab import Vocab
from final.evaluate import compute_metrics, load_model_from_checkpoint
from final.models.caption_model import build_model

REFS = [
    ["a dog runs on the grass", "the dog is running outside"],
    ["a girl in a red dress", "a young girl wearing red"],
]


def test_hyp_trung_khop_diem_cao_hon_hyp_sai():
    perfect = ["a dog runs on the grass", "a girl in a red dress"]
    wrong = ["the cat sleeps indoors quietly", "a boy in blue pants"]
    m_perfect = compute_metrics(perfect, REFS)
    m_wrong = compute_metrics(wrong, REFS)
    assert m_perfect["bleu4"] > 0.99
    assert m_perfect["bleu1"] > m_wrong["bleu1"]
    assert m_perfect["cider"] > m_wrong["cider"]


def test_metric_du_key_va_trong_khoang():
    m = compute_metrics(["a dog runs"], [["a dog runs fast today"]])
    for k in ("bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "cider"):
        assert k in m and m[k] >= 0.0
    assert m["bleu1"] <= 1.0


def test_load_model_from_checkpoint_bo_qua_duong_dan_may_cu(tmp_path):
    """Checkpoint rsync từ máy khác (vd. trainbox) mang theo config với
    đường dẫn tuyệt đối của máy đó (/nonexistent/...). load_model_from_checkpoint
    phải bỏ qua các đường dẫn này và luôn dùng đường dẫn của máy hiện tại,
    nếu không evaluate/visualize local sẽ crash vì không tìm thấy file."""
    real_cfg = Config(decoder="lstm", d_model=16, attn_dim=16, num_layers=1,
                      num_heads=2, ffn_dim=32, feat_dim=8, num_regions=4)
    vocab = Vocab.load(real_cfg.vocab_path)
    model = build_model(real_cfg, len(vocab))

    fake_saved = real_cfg.to_dict()
    fake_saved.update(
        vocab_path="/nonexistent/vocab.json",
        data_root="/nonexistent/data",
        dataset_dir="/nonexistent/data/flickr8k",
        output_dir="/nonexistent/outputs",
        checkpoint_dir="/nonexistent/checkpoints",
    )
    ckpt_path = tmp_path / "fake.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": fake_saved,
        "vocab_size": len(vocab),
        "epoch": 1,
        "val_loss": 1.0,
    }, ckpt_path)

    loaded_model, cfg, loaded_vocab = load_model_from_checkpoint(ckpt_path, "cpu")

    fresh = Config()
    assert cfg.vocab_path == fresh.vocab_path
    assert cfg.data_root == fresh.data_root
    assert cfg.dataset_dir == fresh.dataset_dir
    assert cfg.output_dir == fresh.output_dir
    assert cfg.checkpoint_dir == fresh.checkpoint_dir
    assert not cfg.vocab_path.startswith("/nonexistent")
    assert not cfg.data_root.startswith("/nonexistent")
    assert len(loaded_vocab) == len(vocab)
    assert loaded_model is not None
