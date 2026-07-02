"""
Day 6 — Dịch máy Anh→Việt bằng Encoder-Decoder (Seq2Seq) + Bahdanau Attention.

Bài toán: cho một câu tiếng Anh, sinh ra câu tiếng Việt tương ứng.

Kiến trúc: Encoder GRU đọc câu nguồn thành một chuỗi hidden states; Decoder GRU
sinh từng từ đích, và ở MỖI bước dùng attention để "nhìn lại" toàn bộ câu nguồn
thay vì chỉ dựa vào một context vector cố định — đó chính là điểm nghẽn của
seq2seq thuần (Sutskever 2014) mà Bahdanau 2015 giải quyết.

Dữ liệu: ~12.6k cặp câu Anh-Việt từ Tatoeba (manythings.org/anki), tự tải về
data/tatoeba_envi/ ở lần chạy đầu.

Chạy:  python day6/01_seq2seq_attention_translation.py
Xem thêm lý thuyết trong day6/README.md.
"""

import math
import os
import random
import re
import time
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # backend không cần GUI — vẽ thẳng ra file .png
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Cấu hình chung
# ---------------------------------------------------------------------------
SEED = 42
DATA_URL = "https://www.manythings.org/anki/vie-eng.zip"
DAY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DAY_DIR, "..", "data", "tatoeba_envi")

MAX_LEN = 12        # bỏ cặp có vế dài hơn 12 token — giữ phần lớn corpus, train nhanh
MIN_FREQ = 2        # từ xuất hiện < 2 lần thay bằng <unk> — vocab gọn đi rất nhiều
BATCH_SIZE = 64
D_EMB = 256         # số chiều vector embedding của một từ
D_HID = 256         # số chiều hidden state của GRU
EPOCHS = 15
LR = 1e-3
TF_RATIO = 0.5      # xác suất dùng teacher forcing tại mỗi bước giải mã
CLIP = 1.0          # chặn norm gradient — RNN rất dễ "nổ" gradient
MAX_DECODE_LEN = 20  # trần số từ sinh ra khi dịch (tránh lặp vô hạn)

# Chỉ số của 4 token đặc biệt — cố định ở đầu mọi từ điển
PAD, SOS, EOS, UNK = 0, 1, 2, 3

random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Bước 1: Dữ liệu — tải, chuẩn hoá, lọc, chia train/val
# ---------------------------------------------------------------------------
def tai_du_lieu():
    """Tải và giải nén corpus Tatoeba nếu chưa có. Trả về đường dẫn vie.txt."""
    os.makedirs(DATA_DIR, exist_ok=True)
    duong_dan = os.path.join(DATA_DIR, "vie.txt")
    if not os.path.exists(duong_dan):
        nen = os.path.join(DATA_DIR, "vie-eng.zip")
        print(f"Đang tải {DATA_URL} ...")
        # manythings.org trả 406 cho User-Agent mặc định của urllib (và cả
        # "Mozilla/5.0" trần) → phải giả một chuỗi UA trình duyệt đầy đủ
        ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        yeu_cau = urllib.request.Request(DATA_URL, headers={"User-Agent": ua})
        with urllib.request.urlopen(yeu_cau) as phan_hoi, open(nen, "wb") as f:
            f.write(phan_hoi.read())
        with zipfile.ZipFile(nen) as z:
            z.extract("vie.txt", DATA_DIR)
    return duong_dan


def chuan_hoa(cau):
    """Chuẩn hoá một câu: NFC + chữ thường + tách dấu câu thành token riêng.

    - NFC gộp ký tự tổ hợp ("e" + dấu mũ + dấu hỏi) về một codepoint duy nhất.
      Quan trọng với tiếng Việt: cùng chữ "ể" có thể được gõ theo 2 cách mã hoá
      khác nhau — nếu không chuẩn hoá, từ điển sẽ coi chúng là 2 từ khác nhau.
    - Tách dấu câu để "chạy!" thành 2 token "chạy" và "!". Không tách dấu nháy
      đơn để giữ nguyên các từ tiếng Anh viết tắt như "don't", "i'm".
    """
    cau = unicodedata.normalize("NFC", cau).lower().strip()
    cau = re.sub(r'([.!?,;:"()])', r" \1 ", cau)
    return re.sub(r"\s+", " ", cau).strip()


def doc_cap_cau(duong_dan):
    """Đọc file TSV (mỗi dòng: EN <tab> VI <tab> ghi_công) → list cặp (en, vi)
    đã chuẩn hoá, chỉ giữ cặp mà cả hai vế đều không quá MAX_LEN token."""
    cap = []
    with open(duong_dan, encoding="utf-8") as f:
        for dong in f:
            cot = dong.rstrip("\n").split("\t")
            if len(cot) < 2:
                continue
            en, vi = chuan_hoa(cot[0]), chuan_hoa(cot[1])
            if 0 < len(en.split()) <= MAX_LEN and 0 < len(vi.split()) <= MAX_LEN:
                cap.append((en, vi))
    return cap


def chia_train_val(cap_cau, ty_le_val=0.1):
    """Chia train/val theo *câu nguồn*, không chia theo cặp.

    Một câu tiếng Anh trong Tatoeba thường có nhiều bản dịch tiếng Việt.
    Nếu chia ngẫu nhiên theo cặp thì cùng một câu EN có thể rơi vào cả train
    lẫn val → lúc đánh giá model đã "thấy" câu đó rồi → BLEU bị thổi phồng.
    Gom mọi bản dịch của một câu EN về cùng một phía sẽ tránh rò rỉ này.
    """
    nhom = defaultdict(list)
    for en, vi in cap_cau:
        nhom[en].append(vi)
    cac_en = sorted(nhom)  # sort trước khi shuffle → kết quả tái lập với cùng seed
    random.shuffle(cac_en)
    n_val = int(len(cac_en) * ty_le_val)
    en_val, en_train = cac_en[:n_val], cac_en[n_val:]
    train = [(en, vi) for en in en_train for vi in nhom[en]]
    # Val chỉ giữ 1 bản dịch tham chiếu mỗi câu cho BLEU đơn giản.
    # (BLEU chuẩn cho phép nhiều tham chiếu — xem bài tập về nhà.)
    val = [(en, nhom[en][0]) for en in en_val]
    return train, val


# ---------------------------------------------------------------------------
# Bước 2: Từ điển và Dataset
# ---------------------------------------------------------------------------
class TuDien:
    """Từ điển word-level: ánh xạ token ↔ chỉ số.

    Xây từ tập train DUY NHẤT — nếu đếm cả val thì thông tin của tập đánh giá
    đã rò rỉ vào model. Tiếng Việt viết tách âm tiết bằng dấu cách sẵn nên
    split theo khoảng trắng là đủ dùng.
    """

    def __init__(self, cac_cau):
        dem = Counter(tok for cau in cac_cau for tok in cau.split())
        self.tu = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.tu += sorted(t for t, c in dem.items() if c >= MIN_FREQ)
        self.chi_so = {t: i for i, t in enumerate(self.tu)}

    def __len__(self):
        return len(self.tu)

    def ma_hoa(self, cau):
        """Câu → list chỉ số, kết thúc bằng <eos> để model học được điểm dừng."""
        return [self.chi_so.get(t, UNK) for t in cau.split()] + [EOS]

    def giai_ma(self, chi_so):
        """List chỉ số → câu; dừng tại <eos>, bỏ qua token đặc biệt."""
        tu = []
        for i in chi_so:
            if i == EOS:
                break
            if i not in (PAD, SOS):
                tu.append(self.tu[i])
        return " ".join(tu)


class DuLieuDich(Dataset):
    """Mỗi phần tử: (chỉ số câu EN + <eos>,  [<sos>] + chỉ số câu VI + <eos>)."""

    def __init__(self, cap_cau, td_en, td_vi):
        self.du_lieu = [
            (td_en.ma_hoa(en), [SOS] + td_vi.ma_hoa(vi)) for en, vi in cap_cau
        ]

    def __len__(self):
        return len(self.du_lieu)

    def __getitem__(self, i):
        src, tgt = self.du_lieu[i]
        return torch.tensor(src), torch.tensor(tgt)


def gop_batch(batch):
    """Các câu trong một batch dài ngắn khác nhau → pad về độ dài câu dài nhất.

    Trả kèm độ dài THẬT của từng câu nguồn (giữ trên CPU — pack_padded_sequence
    yêu cầu vậy) để encoder bỏ qua phần pad thay vì "đọc" cả những ô rỗng.
    """
    src, tgt = zip(*batch)
    do_dai_src = torch.tensor([len(s) for s in src])
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD)
    return src, do_dai_src, tgt


# ---------------------------------------------------------------------------
# Bước 3: Model — Encoder, Attention, Decoder
# ---------------------------------------------------------------------------
class BoMaHoa(nn.Module):
    """Encoder: đọc câu nguồn, trả về hidden state tại MỌI vị trí (để attention
    nhìn lại) và hidden cuối cùng (để khởi tạo decoder)."""

    def __init__(self, n_tu):
        super().__init__()
        self.emb = nn.Embedding(n_tu, D_EMB, padding_idx=PAD)
        self.gru = nn.GRU(D_EMB, D_HID, batch_first=True)

    def forward(self, src, do_dai):
        # Pack để GRU chỉ chạy trên token thật, bỏ qua các ô <pad>. Nhờ vậy
        # "hidden cuối" đúng là hidden tại token thật cuối cùng của từng câu,
        # không phải hidden sau khi đã "đọc" thêm một chuỗi pad vô nghĩa.
        emb = self.emb(src)                                    # (B, S, D_EMB)
        goi = nn.utils.rnn.pack_padded_sequence(
            emb, do_dai.cpu(), batch_first=True, enforce_sorted=False)
        ra_goi, hidden = self.gru(goi)
        h_enc, _ = nn.utils.rnn.pad_packed_sequence(
            ra_goi, batch_first=True, padding_value=0.0)
        return h_enc, hidden                     # (B, S, D_HID), (1, B, D_HID)


class ChuYBahdanau(nn.Module):
    """Attention cộng tính (Bahdanau 2015).

    Tại mỗi bước giải mã, chấm điểm mức liên quan giữa trạng thái decoder s
    và hidden h_i của TỪNG vị trí câu nguồn:

        score(h_i, s) = vᵀ · tanh(W_h·h_i + W_s·s)     (một MLP 1 lớp ẩn nhỏ)
        α = softmax(score)                              (trọng số, tổng = 1)
        context c = Σ α_i · h_i                         (tóm tắt CÓ TRỌNG TÂM)

    α cho biết decoder đang "nhìn" vào từ nguồn nào — chính là thứ ta vẽ
    thành heatmap ở cuối bài.
    """

    def __init__(self):
        super().__init__()
        self.W_h = nn.Linear(D_HID, D_HID, bias=False)  # chiếu hidden encoder
        self.W_s = nn.Linear(D_HID, D_HID, bias=False)  # chiếu hidden decoder
        self.v = nn.Linear(D_HID, 1, bias=False)        # nén vector điểm về 1 số

    def forward(self, s, h_enc, mask):
        # s: (B, D_HID) — hidden decoder bước trước
        # h_enc: (B, S, D_HID); mask: (B, S) — True tại token thật
        # unsqueeze để broadcast: W_s·s (B, 1, D_HID) cộng được với mọi vị trí i
        diem = self.v(torch.tanh(self.W_h(h_enc) + self.W_s(s).unsqueeze(1)))
        diem = diem.squeeze(-1)                            # (B, S)
        # Vị trí <pad> nhận điểm -inf → sau softmax trọng số đúng bằng 0,
        # decoder không bao giờ "chú ý" vào ô rỗng.
        diem = diem.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(diem, dim=1)                 # (B, S)
        # Tổng có trọng số các hidden encoder: (B,1,S) @ (B,S,D_HID) → (B,D_HID)
        boi_canh = torch.bmm(alpha.unsqueeze(1), h_enc).squeeze(1)
        return boi_canh, alpha


class BoGiaiMa(nn.Module):
    """Decoder: mỗi bước nhận (từ vừa sinh, context từ attention) → GRU → logits.

    Khác decoder seq2seq thuần: đầu vào GRU được nối thêm context vector c_t
    tính MỚI ở mỗi bước, nên thông tin câu nguồn không phải chen chúc trong
    một vector cố định duy nhất nữa.
    """

    def __init__(self, n_tu):
        super().__init__()
        self.emb = nn.Embedding(n_tu, D_EMB, padding_idx=PAD)
        self.chu_y = ChuYBahdanau()
        self.gru = nn.GRU(D_EMB + D_HID, D_HID, batch_first=True)
        self.ra = nn.Linear(D_HID, n_tu)

    def forward(self, y_truoc, hidden, h_enc, mask):
        # y_truoc: (B,) — token sinh ở bước trước; hidden: (1, B, D_HID)
        boi_canh, alpha = self.chu_y(hidden[-1], h_enc, mask)
        vao = torch.cat([self.emb(y_truoc), boi_canh], dim=1)  # (B, D_EMB+D_HID)
        ra, hidden = self.gru(vao.unsqueeze(1), hidden)        # chạy đúng 1 bước
        return self.ra(ra.squeeze(1)), hidden, alpha           # (B, n_tu), …


class MoHinhDich(nn.Module):
    """Ghép encoder + decoder thành model dịch hoàn chỉnh cho lúc train."""

    def __init__(self, n_tu_en, n_tu_vi):
        super().__init__()
        self.ma_hoa = BoMaHoa(n_tu_en)
        self.giai_ma = BoGiaiMa(n_tu_vi)

    def forward(self, src, do_dai_src, tgt, tf_ratio):
        """Giải mã T-1 bước; tgt có dạng [<sos>, y_1, …, y_n, <eos>, <pad>…].

        Teacher forcing: với xác suất tf_ratio, đầu vào bước t là từ đích THẬT
        (tgt[:, t]) thay vì từ model vừa đoán. Giúp hội tụ nhanh vì lỗi ở một
        bước không kéo sập cả chuỗi phía sau — nhưng tạo "exposure bias":
        lúc dịch thật model phải tự ăn output của chính mình, điều nó chưa
        từng quen nếu train với teacher forcing 100%.
        """
        h_enc, hidden = self.ma_hoa(src, do_dai_src)
        mask = src != PAD
        cac_logits = []
        y = tgt[:, 0]                                # cột đầu luôn là <sos>
        for t in range(1, tgt.size(1)):
            logits, hidden, _ = self.giai_ma(y, hidden, h_enc, mask)
            cac_logits.append(logits)
            dung_that = random.random() < tf_ratio
            y = tgt[:, t] if dung_that else logits.argmax(dim=1)
        return torch.stack(cac_logits, dim=1)        # (B, T-1, n_tu_vi)


if __name__ == "__main__":
    duong_dan = tai_du_lieu()
    train, val = chia_train_val(doc_cap_cau(duong_dan))
    td_en = TuDien(en for en, _ in train)
    td_vi = TuDien(vi for _, vi in train)
    loader = DataLoader(DuLieuDich(train, td_en, td_vi), batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=gop_batch)
    src, do_dai, tgt = next(iter(loader))
    model = MoHinhDich(len(td_en), len(td_vi)).to(DEVICE)
    logits = model(src.to(DEVICE), do_dai, tgt.to(DEVICE), tf_ratio=0.5)
    print(f"Thiết bị: {DEVICE}")
    print(f"logits: {tuple(logits.shape)}  (kỳ vọng ({src.size(0)}, {tgt.size(1) - 1}, {len(td_vi)}))")
    assert logits.shape == (src.size(0), tgt.size(1) - 1, len(td_vi))
    h_enc, hidden = model.ma_hoa(src.to(DEVICE), do_dai)
    _, alpha = model.giai_ma.chu_y(hidden[-1], h_enc, (src != PAD).to(DEVICE))
    print(f"tổng alpha (phải ≈1): {[round(x, 4) for x in alpha.sum(dim=1)[:3].tolist()]}")
    n = sum(p.numel() for p in model.parameters())
    print(f"Số tham số: {n:,}")
