# references:
# https://github.com/mk-minchul/AdaFace/blob/master/head.py
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
# NOTE: most speaker embedding repos use smaller s and m

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AdaFace(nn.Module):
    def __init__(self, m: float = 0.4, h: float = 0.333, s: float = 64.0) -> None:
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.register_buffer("norm_mean", torch.tensor(20.0))
        self.register_buffer("norm_std", torch.tensor(100.0))

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        norms = norms.clip(0.001, 100)
        std, mean = torch.std_mean(norms)
        self.norm_mean.lerp_(mean, 1e-2)
        self.norm_std.lerp_(std, 1e-2)

        margin_scaler = (norms - self.norm_mean) / (self.norm_std + 1e-3)
        margin_scaler = (margin_scaler * self.h).clip(-1.0, 1.0)

        positives = logits[torch.arange(logits.shape[0]), labels]
        theta = positives.clamp(-0.999, 0.999).acos() - self.m * margin_scaler  # g_angle
        positives = theta.clip(0, torch.pi).cos() - self.m * (1.0 + margin_scaler)  # g_add
        logits[torch.arange(logits.shape[0]), labels] = positives
        return F.cross_entropy(logits * self.s, labels)


class ArcFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.5) -> None:
        super().__init__()
        self.m = m
        self.s = s

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        positives = logits[torch.arange(logits.shape[0]), labels]
        theta = positives.clamp(-0.999, 0.999).acos() + self.m
        positives = theta.clip(0.0, torch.pi).cos()
        logits[torch.arange(logits.shape[0]), labels] = positives
        return F.cross_entropy(logits * self.s, labels)


class CosFace(nn.Module):
    def __init__(self, s: float = 64.0, m: float = 0.4) -> None:
        super().__init__()
        self.m = m
        self.s = s

    def forward(self, logits: Tensor, norms: Tensor, labels: Tensor) -> Tensor:
        logits[torch.arange(logits.shape[0]), labels] -= self.m
        return F.cross_entropy(logits * self.s, labels)
