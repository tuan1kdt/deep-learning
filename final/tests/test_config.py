"""Config là hợp đồng trung tâm — test khóa các mặc định mà mọi task sau dựa vào."""
from final.config import Config


def test_defaults_theo_spec():
    cfg = Config()
    assert cfg.decoder == "lstm"
    assert cfg.use_attention is True
    assert cfg.d_model == 512
    assert cfg.max_words == 20
    assert cfg.max_len == 22          # bos + 20 từ + eos
    assert cfg.min_word_freq == 5
    assert cfg.seed == 42
    assert cfg.label_smoothing == 0.1


def test_lr_va_warmup_tu_dong_theo_decoder():
    # Để lr=0 nghĩa là "auto": lstm 3e-4 không warmup, transformer 1e-4 + 2000 step
    assert Config(decoder="lstm").lr == 3e-4
    assert Config(decoder="lstm").warmup_steps == 0
    assert Config(decoder="transformer").lr == 1e-4
    assert Config(decoder="transformer").warmup_steps == 2000
    # Người dùng chỉ định tay thì giữ nguyên
    assert Config(decoder="transformer", lr=5e-4, warmup_steps=100).lr == 5e-4


def test_run_name_tu_dong():
    assert Config(decoder="lstm").run_name == "lstm"
    assert Config(decoder="lstm", use_attention=False).run_name == "lstm_noattn"
    assert Config(decoder="transformer").run_name == "transformer"
    assert Config(run_name="custom").run_name == "custom"


def test_duong_dan_artifact():
    cfg = Config()
    assert cfg.features_path("train").name == "features_train.pt"
    assert cfg.captions_path("test").name == "captions_test.json"


def test_decoder_khong_hop_le_bi_chan():
    import pytest
    with pytest.raises(AssertionError):
        Config(decoder="rnn")
