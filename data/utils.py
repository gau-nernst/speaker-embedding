import torchaudio
from torchaudio.transforms import Resample

SAMPLE_RATE = 16_000
_RESAMPLERS = dict()
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a")


def load_audio(data: str | bytes):
    audio, fs = torchaudio.load(data)
    if fs != SAMPLE_RATE:
        if fs not in _RESAMPLERS:
            # default resampler. might not be good enough
            _RESAMPLERS[fs] = Resample(fs, SAMPLE_RATE)
        audio = _RESAMPLERS[fs](audio)
    return audio.mean(0)
