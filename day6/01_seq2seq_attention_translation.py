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


# ---------------------------------------------------------------------------
# Bước 4: Huấn luyện
# ---------------------------------------------------------------------------
def chay_epoch(model, loader, ham_loss, optimizer=None, tf_ratio=TF_RATIO):
    """Chạy trọn 1 epoch. Có optimizer → train; không có → đánh giá (val).

    Val dùng teacher forcing 100%: loss khi đó đo "khả năng đoán từ tiếp theo
    khi biết toàn bộ tiền tố thật" — ổn định và so sánh được giữa các epoch,
    không phụ thuộc vào chuỗi lỗi tự sinh của model.
    """
    dang_train = optimizer is not None
    model.train(dang_train)
    tong_loss, so_buoc = 0.0, 0
    with torch.set_grad_enabled(dang_train):
        for src, do_dai_src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits = model(src, do_dai_src, tgt, tf_ratio if dang_train else 1.0)
            # Gộp mọi bước thời gian của cả batch thành một batch phẳng cho
            # cross-entropy; ignore_index=PAD → ô pad không đóng góp vào loss.
            loss = ham_loss(logits.reshape(-1, logits.size(-1)),
                            tgt[:, 1:].reshape(-1))
            if dang_train:
                optimizer.zero_grad()
                loss.backward()
                # Chặn norm gradient: một batch "xấu" có thể tạo gradient khổng
                # lồ làm hỏng cả quá trình train (exploding gradient của RNN).
                nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
            tong_loss += loss.item()
            so_buoc += 1
    return tong_loss / so_buoc


# ---------------------------------------------------------------------------
# Bước 5: Suy luận (greedy decoding) và đánh giá BLEU
# ---------------------------------------------------------------------------
@torch.no_grad()
def dich(model, cau_en, td_en, td_vi):
    """Dịch 1 câu bằng greedy decoding: mỗi bước chọn từ có logit cao nhất.

    Greedy đơn giản nhưng tham lam — chọn sai một từ là không quay lại được.
    Beam search giữ k ứng viên tốt nhất mỗi bước sẽ dịch tốt hơn (bài tập).
    Trả về (list token đích, ma trận attention (T_ra, S) để vẽ heatmap).
    """
    model.eval()
    src = torch.tensor([td_en.ma_hoa(chuan_hoa(cau_en))], device=DEVICE)
    do_dai = torch.tensor([src.size(1)])
    h_enc, hidden = model.ma_hoa(src, do_dai)
    mask = src != PAD
    y = torch.tensor([SOS], device=DEVICE)
    tu_ra, cac_alpha = [], []
    for _ in range(MAX_DECODE_LEN):
        logits, hidden, alpha = model.giai_ma(y, hidden, h_enc, mask)
        y = logits.argmax(dim=1)
        cac_alpha.append(alpha.squeeze(0).cpu())
        if y.item() == EOS:
            tu_ra.append("<eos>")
            break
        tu_ra.append(td_vi.tu[y.item()])
    return tu_ra, torch.stack(cac_alpha).numpy()


def n_gram(tu, n):
    return [tuple(tu[i:i + n]) for i in range(len(tu) - n + 1)]


def bleu_corpus(cac_ref, cac_hyp):
    """BLEU-4 mức corpus (Papineni 2002), không smoothing.

    - Precision bị "clip": mỗi n-gram trong bản dịch máy chỉ được tính tối đa
      bằng số lần nó có mặt trong tham chiếu — chặn kiểu ăn gian lặp từ
      ("the the the" không được 3 điểm unigram "the").
    - Brevity penalty phạt bản dịch ngắn hơn tham chiếu, vì precision thuần
      không phạt việc dịch thiếu (câu 2 từ đều đúng → precision 100%!).
    - Cộng dồn số đếm trên TOÀN corpus rồi mới tính (corpus-level), không
      lấy trung bình BLEU từng câu.
    """
    khop = [0] * 4
    tong = [0] * 4
    len_ref = len_hyp = 0
    for ref, hyp in zip(cac_ref, cac_hyp):
        len_ref += len(ref)
        len_hyp += len(hyp)
        for n in range(1, 5):
            dem_ref = Counter(n_gram(ref, n))
            dem_hyp = Counter(n_gram(hyp, n))
            khop[n - 1] += sum(min(c, dem_ref[g]) for g, c in dem_hyp.items())
            tong[n - 1] += max(len(hyp) - n + 1, 0)
    if min(khop) == 0:  # không khớp nổi n-gram nào ở một bậc → BLEU = 0
        return 0.0
    log_p = sum(math.log(k / t) for k, t in zip(khop, tong)) / 4
    bp = 1.0 if len_hyp > len_ref else math.exp(1 - len_ref / len_hyp)
    return 100 * bp * math.exp(log_p)


# ---------------------------------------------------------------------------
# Bước 6: Vẽ kết quả
# ---------------------------------------------------------------------------
def ve_loss(loss_train, loss_val):
    fig, ax = plt.subplots(figsize=(8, 5))
    cac_epoch = range(1, len(loss_train) + 1)
    ax.plot(cac_epoch, loss_train, color="#1f77b4", linewidth=2, label="train")
    ax.plot(cac_epoch, loss_val, color="#d95f02", linewidth=2, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Seq2Seq + Attention — training loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(DAY_DIR, "seq2seq_loss.png")
    fig.savefig(out, dpi=120)
    print(f"Đã lưu {out}")


def ve_heatmap(model, cac_cau, td_en, td_vi):
    """Heatmap attention cho vài câu val: trục x = token nguồn (EN), trục y =
    token đích model sinh ra (VI); ô càng đậm = trọng số attention càng lớn.

    Với cặp Anh-Việt trật tự từ khá giống nhau nên kỳ vọng thấy một "đường
    chéo" mờ — mỗi từ đích chủ yếu nhìn vào từ nguồn cùng vị trí tương đối.
    """
    # layout="constrained" tự chừa chỗ cho colorbar chung và tick label xoay 45°
    fig, cac_ax = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for i, (ax, (en, _)) in enumerate(zip(cac_ax.flat, cac_cau)):
        tu_ra, alpha = dich(model, en, td_en, td_vi)
        tu_vao = chuan_hoa(en).split() + ["<eos>"]
        im = ax.imshow(alpha, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(tu_vao)), tu_vao, rotation=45, ha="right")
        ax.set_yticks(range(len(tu_ra)), tu_ra)
        # 4 ô cùng ý nghĩa trục → chỉ ghi nhãn ở mép ngoài cho đỡ rối
        if i >= 2:
            ax.set_xlabel("câu nguồn (EN)")
        if i % 2 == 0:
            ax.set_ylabel("câu sinh ra (VI)")
    fig.colorbar(im, ax=[a for a in cac_ax.flat], label="trọng số attention α",
                 shrink=0.8)
    fig.suptitle("Decoder 'nhìn' vào từ nguồn nào khi sinh từng từ đích?")
    out = os.path.join(DAY_DIR, "attention_heatmap.png")
    fig.savefig(out, dpi=120)
    print(f"Đã lưu {out}")


# ---------------------------------------------------------------------------
# Ghép tất cả lại
# ---------------------------------------------------------------------------
def main():
    print(f"Thiết bị: {DEVICE}")

    # --- Dữ liệu ---
    duong_dan = tai_du_lieu()
    cap_cau = doc_cap_cau(duong_dan)
    train, val = chia_train_val(cap_cau)
    td_en = TuDien(en for en, _ in train)
    td_vi = TuDien(vi for _, vi in train)
    print(f"Cặp câu sau lọc: {len(cap_cau)} (train {len(train)}, val {len(val)})")
    print(f"Vocab EN: {len(td_en)} | Vocab VI: {len(td_vi)}")

    loader_train = DataLoader(DuLieuDich(train, td_en, td_vi),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=gop_batch)
    loader_val = DataLoader(DuLieuDich(val, td_en, td_vi),
                            batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=gop_batch)

    # --- Model & huấn luyện ---
    model = MoHinhDich(len(td_en), len(td_vi)).to(DEVICE)
    print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")
    ham_loss = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    lich_su_train, lich_su_val = [], []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss_train = chay_epoch(model, loader_train, ham_loss, optimizer)
        loss_val = chay_epoch(model, loader_val, ham_loss)
        lich_su_train.append(loss_train)
        lich_su_val.append(loss_val)
        print(f"Epoch {epoch:2d}/{EPOCHS} | loss train {loss_train:.3f} "
              f"| loss val {loss_val:.3f} | {time.time() - t0:.0f}s")

    # --- BLEU trên tập val: toàn câu model CHƯA TỪNG thấy lúc train ---
    cac_ref, cac_hyp = [], []
    for en, vi in val:
        tu_ra, _ = dich(model, en, td_en, td_vi)
        cac_hyp.append([t for t in tu_ra if t != "<eos>"])
        cac_ref.append(vi.split())
    print(f"\nBLEU-4 trên {len(val)} câu val: {bleu_corpus(cac_ref, cac_hyp):.1f}")

    # --- Dịch mẫu để xem chất lượng bằng mắt ---
    print("\nMột vài câu dịch mẫu (từ tập val):")
    for en, vi in val[:10]:
        tu_ra, _ = dich(model, en, td_en, td_vi)
        may = " ".join(t for t in tu_ra if t != "<eos>")
        print(f"  EN : {en}\n  REF: {vi}\n  MÁY: {may}\n")

    # --- Vẽ ---
    ve_loss(lich_su_train, lich_su_val)
    # Chọn câu val độ dài vừa phải cho heatmap dễ đọc
    cau_heatmap = [c for c in val if 4 <= len(c[0].split()) <= 8][:4]
    ve_heatmap(model, cau_heatmap, td_en, td_vi)


if __name__ == "__main__":
    main()
