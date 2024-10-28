from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from . import transforms as T
from .utils import AUDIO_EXTENSIONS, load_audio


def cycle(dloader: DataLoader):
    epoch_idx = 0
    while True:
        if hasattr(dloader.sampler, "set_epoch"):
            dloader.sampler.set_epoch(epoch_idx)
        yield from dloader
        epoch_idx += 1


def build_train_dloader(dataset: str, batch_size: int, augmentations: list[str], n_workers: int = 4, **kwargs):
    transform = nn.Sequential()
    for aug in augmentations:
        transform.append(eval(aug, dict(T=T)))

    if dataset.startswith("wds://"):
        import webdataset as wds

        audio_key = kwargs.pop("audio_key")

        def preprocess(sample: dict):
            audio = load_audio(sample[audio_key])
            audio = transform(audio)
            label = int(sample["cls"].decode())
            return audio, label

        ds = wds.WebDataset(dataset.removeprefix("wds://"), shardshuffle=True, nodesplitter=wds.split_by_node)
        ds = ds.map(preprocess)

        dloader = DataLoader(ds, batch_size, num_workers=n_workers, pin_memory=True)
        size = float("inf")

    else:
        ds = DatasetFolder(
            root=dataset,
            loader=load_audio,
            extensions=AUDIO_EXTENSIONS,
            transform=transform,
        )
        dloader = DataLoader(ds, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True)
        size = len(ds)

    return cycle(dloader), size
