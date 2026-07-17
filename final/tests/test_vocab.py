from final.data.vocab import BOS_ID, EOS_ID, PAD_ID, UNK_ID, Vocab, build_vocab, tokenize


def test_tokenize_lowercase_bo_dau_cau():
    assert tokenize("A dog, RUNS fast!") == ["a", "dog", "runs", "fast"]
    assert tokenize("  nhiều   khoảng  trắng ") == ["nhiều", "khoảng", "trắng"]


def test_special_ids_co_dinh():
    assert (PAD_ID, BOS_ID, EOS_ID, UNK_ID) == (0, 1, 2, 3)


def _vocab_nho():
    caps = ["a dog runs", "a dog sleeps", "a cat runs"]  # dog/a/runs xuất hiện ≥2
    return build_vocab(caps, min_freq=2)


def test_min_freq_loc_tu_hiem():
    v = _vocab_nho()
    assert "dog" in v.word2id and "a" in v.word2id and "runs" in v.word2id
    assert "sleeps" not in v.word2id and "cat" not in v.word2id


def test_encode_unk_va_truncate():
    v = _vocab_nho()
    ids = v.encode("a cat runs", max_words=2)      # cat → UNK, cắt còn 2 từ
    assert len(ids) == 2
    assert ids[1] == UNK_ID


def test_roundtrip_decode_dung_o_eos():
    v = _vocab_nho()
    ids = v.encode("a dog runs", max_words=20)
    full = ids + [EOS_ID, v.word2id["dog"], PAD_ID]   # rác sau EOS phải bị bỏ
    assert v.decode(full) == "a dog runs"


def test_save_load_giu_nguyen(tmp_path):
    v = _vocab_nho()
    p = tmp_path / "vocab.json"
    v.save(p)
    v2 = Vocab.load(p)
    assert v2.word2id == v.word2id and len(v2) == len(v)
