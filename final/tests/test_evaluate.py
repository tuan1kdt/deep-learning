from final.evaluate import compute_metrics

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
    for k in ("bleu1", "bleu2", "bleu3", "bleu4", "cider"):
        assert k in m and m[k] >= 0.0
    assert m["bleu1"] <= 1.0
