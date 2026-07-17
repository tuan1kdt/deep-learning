"""Greedy và beam search trên interface chung (feats, cap_in) → (logits, attn).

Cả hai decoder đều decode bằng cách chạy lại toàn bộ prefix mỗi bước — O(T²)
nhưng T ≤ 22 nên không đáng kể, đổi lại một code path duy nhất cho cả LSTM
lẫn Transformer (không cần incremental state riêng cho từng loại).
"""
import torch

from final.data.vocab import BOS_ID, EOS_ID


def _trim(ids: list[int]) -> list[int]:
    """Cắt tại EOS đầu tiên, bỏ BOS — trả về đúng chuỗi từ."""
    out = []
    for i in ids[1:]:                      # bỏ BOS ở đầu
        if i == EOS_ID:
            break
        out.append(i)
    return out


@torch.no_grad()
def greedy_decode(model, feats, max_len: int):
    """feats (B,49,2048) → (list B chuỗi id đã trim, attn (B,T_sinh,R))."""
    model.eval()
    B = feats.size(0)
    cap = torch.full((B, 1), BOS_ID, dtype=torch.long, device=feats.device)
    for _ in range(max_len - 1):
        logits, attn = model(feats, cap)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        cap = torch.cat([cap, nxt], dim=1)
        if (cap == EOS_ID).any(dim=1).all():
            break
    seqs = [_trim(row.tolist()) for row in cap]
    return seqs, attn


@torch.no_grad()
def beam_search(model, feats_one, beam_size: int, max_len: int) -> list[int]:
    """Beam search cho MỘT ảnh, chấm theo log-prob trung bình mỗi token
    (length normalization — không phạt câu dài một cách mù quáng)."""
    model.eval()
    device = feats_one.device
    beams = [([BOS_ID], 0.0)]              # (chuỗi id, tổng log-prob)
    finished = []
    for _ in range(max_len - 1):
        alive = [b for b in beams if b[0][-1] != EOS_ID]
        finished += [b for b in beams if b[0][-1] == EOS_ID]
        if not alive:
            break
        cap = torch.tensor([b[0] for b in alive], dtype=torch.long, device=device)
        feats = feats_one.expand(len(alive), -1, -1)
        logits, _ = model(feats, cap)
        logprobs = logits[:, -1].log_softmax(dim=-1)       # (n_alive, V)
        top_lp, top_id = logprobs.topk(beam_size, dim=-1)
        candidates = [
            (alive[i][0] + [top_id[i, k].item()], alive[i][1] + top_lp[i, k].item())
            for i in range(len(alive)) for k in range(beam_size)
        ]
        beams = sorted(candidates, key=lambda b: b[1], reverse=True)[:beam_size]
    finished += beams
    # điểm = log-prob trung bình trên số token đã sinh (không tính BOS)
    best = max(finished, key=lambda b: b[1] / max(1, len(b[0]) - 1))
    return _trim(best[0])
