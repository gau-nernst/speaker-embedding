import timm
from torch import Tensor, nn

from .transformer import AudioTransformer


class SpeakerEmbedder(nn.Module):
    def __init__(self, backbone: str, embed_dim: int = 512, backbone_kwargs: dict | None = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or dict()
        if backbone.startswith("audio_transformer_"):
            self.backbone = AudioTransformer.from_config(backbone, n_classes=embed_dim, **backbone_kwargs)
            self.is_timm = False
        else:
            self.backbone = timm.create_model(backbone, in_chans=1, num_classes=embed_dim, **backbone_kwargs)
            self.is_timm = True

    def forward(self, x: Tensor) -> Tensor:
        if self.is_timm:
            x = x.unsqueeze(-3)  # convert to 2D spectrogram (*, 1, n_mels, n_steps)
        x = self.backbone(x)
        return x
