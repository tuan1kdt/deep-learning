"""extract_captions là logic thuần — test bằng dict giả, không cần network."""
from final.data.download import extract_captions


def test_extract_captions_giu_thu_tu_va_du_5_cau():
    rows = [
        {"image": None, "caption_0": "A dog runs.", "caption_1": "b", "caption_2": "c",
         "caption_3": "d", "caption_4": "e"},
        {"image": None, "caption_0": "x", "caption_1": "y", "caption_2": "z",
         "caption_3": "w", "caption_4": "v"},
    ]
    caps = extract_captions(rows)
    assert len(caps) == 2
    assert caps[0] == ["A dog runs.", "b", "c", "d", "e"]
    assert caps[1][4] == "v"


def test_extract_captions_schema_flickr30k_cot_caption_la_list():
    rows = [
        {"image": None, "caption": ["a", "b", "c", "d", "e"]},
        {"image": None, "caption": ["f", "g", "h", "i", "j"]},
    ]
    caps = extract_captions(rows)
    assert caps == [["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"]]
