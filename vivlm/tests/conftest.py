import pytest
from vivlm.config import SPECIAL_TOKENS

SAMPLE_VI = (
    "Ba chiếc thuyền đang di chuyển trên dòng sông rộng lớn. "
    "Một cậu bé đá quả bóng trong sân cỏ xanh mướt. "
    "Người phụ nữ bán rau ở chợ nổi miền Tây sông nước. "
) * 30


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    from vivlm.tokenizer_train import train_bpe
    d = tmp_path_factory.mktemp("tok")
    txt = d / "sample.txt"
    txt.write_text(SAMPLE_VI, encoding="utf-8")
    return train_bpe([str(txt)], str(d / "tokenizer.json"), vocab_size=300)
