import math

import torch
from torch import Tensor, nn

from .utils import SAMPLE_RATE


def randint(low: int, high: int):
    return torch.randint(low, high, size=()).item()


class CropAudio(nn.Module):
    def __init__(self, duration: float, random: bool = False) -> None:
        super().__init__()
        self.target_len = int(duration * SAMPLE_RATE)
        self.random = random

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.target_len:
            n_repeat = math.ceil(self.target_len / x.shape[-1])
            x = x.repeat((1,) * (x.ndim - 1) + (n_repeat,))

        start_idx = randint(0, x.shape[-1] - self.target_len) if self.random else 0
        return x[..., start_idx : start_idx + self.target_len]
