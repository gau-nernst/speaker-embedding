import timm
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram


class LogMelspectrogram(MelSpectrogram):
    def __init__(self, n_mels: int = 80) -> None:
        super().__init__(
            sample_rate=16_000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            mel_scale="slaney",
            norm="slaney",
        )

    def forward(self, x: Tensor) -> Tensor:
        melspec = super().forward(x)[..., :-1]
        logspec = melspec.clip(1e-10).log()
        logspec = logspec - logspec.mean(-1, keepdim=True)  # cmn
        return logspec  # (n_mels, n_frames)


class TimmAudioBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_mels: int = 80,
        out_dim: int = 256,
        activation_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.logmelspec = LogMelspectrogram(n_mels)
        self.backbone = timm.create_model(model_name, in_chans=1, num_classes=out_dim, **kwargs)
        if activation_checkpointing:
            self.backbone.set_gradient_checkpointing()

    def forward(self, x: Tensor) -> Tensor:
        x = self.logmelspec(x).unsqueeze(-3)  # (B, 1, n_mels, n_frames)
        x = self.backbone(x)
        return x
