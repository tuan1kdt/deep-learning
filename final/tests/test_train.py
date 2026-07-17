import torch
from torch import nn

from final.data.vocab import PAD_ID
from final.train import build_scheduler, make_criterion


def test_scheduler_warmup_tang_roi_cosine_giam():
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]                      # đang warmup: tăng dần
    assert abs(lrs[9] - 1.0) < 0.15             # đỉnh ~ base lr
    assert lrs[50] > lrs[99]                    # sau đó cosine giảm
    assert lrs[99] >= 0.0


def test_scheduler_khong_warmup_bat_dau_tu_dinh():
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1.0)
    build_scheduler(opt, warmup_steps=0, total_steps=100)
    assert abs(opt.param_groups[0]["lr"] - 1.0) < 1e-6


def test_criterion_bo_qua_pad():
    crit = make_criterion(label_smoothing=0.0)
    logits = torch.randn(2, 3, 10)
    target = torch.tensor([[4, PAD_ID, PAD_ID], [4, 5, PAD_ID]])
    loss_it = crit(logits.reshape(-1, 10), target.reshape(-1))
    # đổi logits tại vị trí pad không được đổi loss
    logits2 = logits.clone()
    logits2[0, 1] += 100.0
    loss2 = crit(logits2.reshape(-1, 10), target.reshape(-1))
    assert torch.allclose(loss_it, loss2)
