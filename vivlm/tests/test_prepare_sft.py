import io
import json
import os
import zipfile

from PIL import Image

from vivlm.data.prepare_sft import _vivqa_rows


def _make_zip(zip_path, entries):
    """entries: list of (arcname, size) -> ghi ảnh PNG giả vào zip."""
    with zipfile.ZipFile(zip_path, "w") as z:
        for arcname, size in entries:
            buf = io.BytesIO()
            Image.new("RGB", size, (10, 20, 30)).save(buf, format="JPEG")
            z.writestr(arcname, buf.getvalue())


def test_vivqa_image_paths_resolve(tmp_path):
    zip_path = tmp_path / "images.zip"
    _make_zip(zip_path, [
        ("COCO_0.jpg", (32, 32)),
        ("sub/COCO_1.jpg", (32, 32)),
    ])
    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps({
        "images": {"0": "COCO_0.jpg", "1": "COCO_1.jpg"},
        "annotations": {
            "a": {"image_id": 0, "question": "Có gì trong ảnh?", "answer": "cái ghế"},
            "b": {"image_id": 1, "question": "Màu gì?", "answer": "màu xanh"},
        },
    }, ensure_ascii=False), encoding="utf-8")

    out_img_dir = str(tmp_path / "sft")
    rows = _vivqa_rows(str(json_path), str(zip_path), out_img_dir,
                       "images/vivqa_train")

    assert len(rows) == 2
    for row in rows:
        # img_root truyền vào loader/eval CHÍNH LÀ out_img_dir — path lưu
        # trong row["image"] phải tương đối với out_img_dir, không phải
        # dirname(out_img_dir) (bug cũ sinh thừa 1 cấp "sft/").
        assert os.path.exists(os.path.join(out_img_dir, row["image"])), (
            f"image path {row['image']!r} không resolve được dưới {out_img_dir}")
