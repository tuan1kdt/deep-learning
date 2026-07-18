import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vivlm.data.prepare_pretrain import BinWriter, tokenize_parquet

DOCS = ["ba chiếc thuyền trên sông", "cậu bé đá bóng", "chợ nổi miền Tây"]


def _make_parquet(path):
    pq.write_table(pa.table({"text": DOCS, "id": ["a", "b", "c"]}), path)


def test_binwriter_uint16(tmp_path):
    p = tmp_path / "t.bin"
    w = BinWriter(str(p))
    w.write([1, 2, 65535])
    w.close()
    arr = np.fromfile(p, dtype=np.uint16)
    assert arr.tolist() == [1, 2, 65535] and w.count == 3


def test_tokenize_parquet_eos_per_doc(tmp_path, tiny_tokenizer):
    pqt = tmp_path / "d.parquet"
    _make_parquet(str(pqt))
    out = tmp_path / "o.bin"
    w = BinWriter(str(out))
    tokenize_parquet(str(pqt), tiny_tokenizer, w, eos_id=0)
    w.close()
    arr = np.fromfile(out, dtype=np.uint16)
    assert (arr == 0).sum() == len(DOCS)          # mỗi doc đúng 1 eos
    # decode lại doc đầu (đến eos đầu tiên) khớp nguyên văn
    first = arr[: np.where(arr == 0)[0][0]].tolist()
    assert tiny_tokenizer.decode(first) == DOCS[0]
