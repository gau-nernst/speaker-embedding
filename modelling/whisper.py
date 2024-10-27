# https://github.com/openai/whisper/blob/main/whisper/model.py

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torchaudio.transforms import MelSpectrogram


class MHA(nn.Module):
    head_dim = 64

    def __init__(self, d_model: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias)
        self.key = nn.Linear(d_model, d_model, bias)
        self.value = nn.Linear(d_model, d_model, bias)
        self.out = nn.Linear(d_model, d_model, bias)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        q = self.query(x).unflatten(-1, (-1, self.head_dim)).transpose(-2, -3)  # (*, n_heads, L, head_dim)
        k = self.key(x).unflatten(-1, (-1, self.head_dim)).transpose(-2, -3)
        v = self.value(x).unflatten(-1, (-1, self.head_dim)).transpose(-2, -3)

        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        return self.out(out.transpose(-2, -3).flatten(-2))


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, in_dim, bias)
        self.dropout = nn.Dropout(dropout)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, bias: bool = True, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, bias, dropout)
        self.mlp_ln = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), bias, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x


def make_buffers_non_persistent(m: nn.Module):
    for name, buffer in m.named_buffers(recurse=False):
        m.register_buffer(name, buffer, persistent=False)


def sinusoids(length: int, channels: int, max_timescale: float = 10_000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


# https://github.com/openai/whisper/blob/main/whisper/audio.py
class WhisperLogMelspec(MelSpectrogram):
    def __init__(self, n_mels: int = 80) -> None:
        super().__init__(
            sample_rate=16_000,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            mel_scale="slaney",
            norm="slaney",
        )
        self.apply(make_buffers_non_persistent)

    def forward(self, x: Tensor) -> Tensor:
        melspec = super().forward(x)[..., :-1]
        logspec = melspec.clip(1e-10).log10()
        logspec = logspec.clip(logspec.amax((-1, -2), keepdim=True) - 8.0)
        logspec = (logspec + 4.0) / 4.0
        return logspec


class WhisperEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        n_layers: int = 12,
        d_model: int = 768,
        out_dim: int = 256,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_duration: float = 30.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.logmelspec = WhisperLogMelspec(n_mels)
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, 1, 1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, 2, 1)
        self.register_buffer("pos_embed", sinusoids(int(16_000 * max_duration), d_model), persistent=False)
        self.blocks = nn.ModuleList(*[EncoderBlock(d_model, bias, mlp_ratio, dropout) for _ in range(n_layers)])
        self.ln_pos = nn.LayerNorm(d_model)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, x: Tensor) -> Tensor:
        x = self.logmelspec(x)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + self.pos_embed).type_as(x)  # pos_embed may have different dtype from x

        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.activation_checkpointing else block(x)

        x = self.ln_pos(x)
        x = x.mean(-2)  # mean pooling
        return x
