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


def test_dataset_flickr30k_di_cay_thu_muc_rieng():
    """Flickr30k phải trỏ vào data30k/ để không đè artifact Flickr8k."""
    cfg = Config(dataset="flickr30k")
    assert cfg.hf_dataset == "nlphuji/flickr30k"
    assert cfg.data_root.endswith("data30k")
    assert cfg.dataset_dir.endswith("data30k/flickr30k")
    assert cfg.vocab_path.endswith("data30k/vocab.json")
    assert cfg.features_path("train").parent.name == "data30k"
    # Mặc định flickr8k giữ nguyên đường dẫn cũ (tương thích artifact đã có)
    old = Config()
    assert old.hf_dataset == "jxie/flickr8k"
    assert old.data_root.endswith("data")
    assert old.dataset_dir.endswith("data/flickr8k")


def test_encoder_r101_doi_ten_file_feature():
    """resnet101 dùng hậu tố _r101; resnet50 giữ tên cũ."""
    assert Config().features_path("train").name == "features_train.pt"
    assert (Config(encoder="resnet101").features_path("train").name
            == "features_train_r101.pt")


def test_dataset_encoder_khong_hop_le_bi_chan():
    import pytest
    with pytest.raises(AssertionError):
        Config(dataset="coco")
    with pytest.raises(AssertionError):
        Config(encoder="vit")
