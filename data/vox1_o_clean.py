from pathlib import Path

import requests
from torch.utils.data import Dataset

from .transforms import CropAudio
from .utils import load_audio


class Vox1OClean(Dataset):
    def __init__(self, data_dir: str, duration: float = 4.0, random_crop: bool = False) -> None:
        super().__init__()
        meta_url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"
        meta_path = Path(__file__).parent / meta_url.split("/")[-1]
        if not meta_path.exists():
            resp = requests.get(meta_url)
            resp.raise_for_status()
            with open(meta_path, "wb") as f:
                f.write(resp.content)

        self.data_dir = Path(data_dir)
        self.trials = [line.rstrip().split() for line in open(meta_path)]
        self.transform = CropAudio(duration, random=random_crop)

    def __getitem__(self, idx: int):
        label, path1, path2 = self.trials[idx]
        audio1 = self.transform(load_audio(self.data_dir / path1))
        audio2 = self.transform(load_audio(self.data_dir / path2))
        return audio1, audio2, int(label)

    def __len__(self):
        return len(self.trials)
