# https://arxiv.org/abs/1706.03762
# https://github.com/openai/whisper/blob/main/whisper/model.py

import torch.nn.functional as F
from torch import Tensor, nn


class MHA(nn.Module):
    head_dim = 64

    def __init__(self, d_model: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias)
        self.k_proj = nn.Linear(d_model, d_model, bias)
        self.v_proj = nn.Linear(d_model, d_model, bias)
        self.out_proj = nn.Linear(d_model, d_model, bias)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        q = self.q_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(-2, -3)  # (*, n_heads, L, head_dim)
        k = self.k_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(-2, -3)
        v = self.v_proj(x).unflatten(-1, (-1, self.head_dim)).transpose(-2, -3)

        dropout = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        return self.out_proj(out.transpose(-2, -3).flatten(-2))


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
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa = MHA(d_model, bias, dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), bias, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.sa_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class AudioTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 80,
        n_classes: int = 512,
        n_layers: int = 12,
        d_model: int = 768,
        bias: bool = True,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # TODO: downsampling?
        self.patch_embed = nn.Conv1d(in_dim, d_model, 3, 1, 1)
        # TODO: positional encodings (sinusoidal and learned?)
        self.blocks = nn.Sequential(*[EncoderBlock(d_model, bias, mlp_ratio, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).transpose(-1, -2)
        x = self.blocks(x)
        x = self.head(self.norm(x).mean(-2))
        return x

    @staticmethod
    def from_config(name: str, **kwargs) -> "AudioTransformer":
        variant = name.removeprefix("audio_transformer_")
        d_model, n_layers = dict(
            tiny=(192, 12),
            small=(384, 12),
            base=(768, 12),
            large=(1024, 24),
        )[variant]
        return AudioTransformer(d_model=d_model, n_layers=n_layers, **kwargs)
