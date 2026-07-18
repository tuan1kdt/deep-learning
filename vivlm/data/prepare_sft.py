"""Chuẩn hóa UIT-ViIC + KTVIC + OpenViVQA về jsonl thống nhất + kho ảnh.

Chạy 1 lần trên trainbox (tải ~1.7GB ảnh + annotations). Schema các nguồn đã
verify 2026-07-18 — xem docstring từng hàm.
"""
import argparse
import hashlib
import json
import os
import zipfile

CAPTION_PROMPTS = [
    "Mô tả bức ảnh.",
    "Ảnh này có gì?",
    "Hãy miêu tả nội dung bức ảnh.",
    "Nói về những gì bạn thấy trong ảnh.",
    "Viết một câu mô tả cho bức ảnh.",
]


def _h(uid):
    return int(hashlib.md5(uid.encode()).hexdigest(), 16)


def _caption_rows(hf_split, source, img_dir, rel_prefix):
    """UIT-ViIC / KTVIC: image_uid + image(PIL) + caption_vi(list[str])."""
    rows, refs = [], []
    for ex in hf_split:
        uid = ex["image_uid"]
        rel = f"images/{source}_{uid}.jpg"
        path = os.path.join(img_dir, f"{source}_{uid}.jpg")
        if not os.path.exists(path):
            ex["image"].convert("RGB").save(path, quality=90)
        caps = [c.strip() for c in ex["caption_vi"] if c.strip()]
        refs.append({"image": rel, "refs": caps})
        for j, c in enumerate(caps):
            rows.append({"image": rel,
                         "prompt": CAPTION_PROMPTS[_h(f"{uid}_{j}") % 5],
                         "response": c, "source": source})
    return rows, refs


def _vivqa_rows(json_path, img_zip, out_img_dir, rel_dirname):
    ann = json.load(open(json_path, encoding="utf-8"))
    dest = os.path.join(out_img_dir, rel_dirname)
    if not os.path.isdir(dest):
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(dest)
    # zip có thể chứa thư mục con — index theo basename
    fmap = {}
    for root, _, files in os.walk(dest):
        for f in files:
            fmap[f] = os.path.relpath(os.path.join(root, f),
                                      os.path.dirname(out_img_dir))
    rows = []
    for a in ann["annotations"].values():
        fname = ann["images"][str(a["image_id"])]
        if fname not in fmap:
            continue
        rows.append({"image": fmap[fname], "prompt": a["question"].strip(),
                     "response": a["answer"].strip(), "source": "vivqa"})
    return rows


def _dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} dòng")


def main():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vivlm/data/sft")
    args = ap.parse_args()
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    viic = load_dataset("ai-enthusiasm-community/UIT-ViIC")
    ktvic = load_dataset("ai-enthusiasm-community/KTVIC")
    tr_viic, tr_viic_refs = _caption_rows(viic["train"], "viic", img_dir,
                                          args.out)
    te_viic_rows, te_viic_refs = _caption_rows(viic["validation"], "viic",
                                               img_dir, args.out)
    tr_ktvic, tr_ktvic_refs = _caption_rows(ktvic["train"], "ktvic",
                                            img_dir, args.out)
    te_ktvic_rows, te_ktvic_refs = _caption_rows(ktvic["test"], "ktvic",
                                                 img_dir, args.out)

    vqa_train = _vivqa_rows(
        hf_hub_download("uitnlp/OpenViVQA-dataset", "vlsp2023_train_data.json",
                        repo_type="dataset"),
        hf_hub_download("uitnlp/OpenViVQA-dataset", "train-images.zip",
                        repo_type="dataset"),
        args.out, "images/vivqa_train")
    vqa_dev = _vivqa_rows(
        hf_hub_download("uitnlp/OpenViVQA-dataset", "vlsp2023_dev_data.json",
                        repo_type="dataset"),
        hf_hub_download("uitnlp/OpenViVQA-dataset", "dev-images.zip",
                        repo_type="dataset"),
        args.out, "images/vivqa_dev")

    all_train = tr_viic + tr_ktvic + vqa_train
    train = [r for r in all_train if _h(r["image"]) % 50 != 0]   # 98%
    val = [r for r in all_train if _h(r["image"]) % 50 == 0]     # 2%
    _dump(os.path.join(args.out, "train.jsonl"), train)
    _dump(os.path.join(args.out, "val.jsonl"), val)
    _dump(os.path.join(args.out, "test_viic.jsonl"), te_viic_refs)
    _dump(os.path.join(args.out, "test_ktvic.jsonl"), te_ktvic_refs)
    # VQA eval từ dev (test VLSP không công bố answer)
    _dump(os.path.join(args.out, "test_vivqa.jsonl"),
          [{"image": r["image"], "prompt": r["prompt"], "refs": [r["response"]]}
           for r in vqa_dev])
    # refs caption theo ảnh cho SCST (chỉ ảnh caption thuộc train split)
    train_imgs = {r["image"] for r in train}
    _dump(os.path.join(args.out, "train_caption_refs.jsonl"),
          [g for g in tr_viic_refs + tr_ktvic_refs if g["image"] in train_imgs])


if __name__ == "__main__":
    main()
