"""Đọc token bin bằng memmap — không load 6GB vào RAM, không bottleneck CPU."""
import numpy as np
import torch


class TokenBin:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size, context, device="cpu", generator=None):
        ix = torch.randint(len(self.data) - context - 1, (batch_size,),
                           generator=generator)
        x = torch.stack([torch.from_numpy(
            self.data[i:i + context].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            self.data[i + 1:i + 1 + context].astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)
