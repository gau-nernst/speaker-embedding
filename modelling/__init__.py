import timm
from torch import Tensor, nn

from .transformer import AudioTransformer


class TimmAudio(nn.Module):
    def __init__(self, backbone: str, embed_dim: int = 512, backbone_kwargs: dict | None = None) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, in_chans=1, num_classes=embed_dim, **(backbone_kwargs or dict()))

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-3)
        x = self.backbone(x)
        return x
