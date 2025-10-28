"""Musdb Contrastive Torch Dataset.

Ported from COCOLA: https://github.com/gladia-research-group/cocola
"""

import os
from typing import Dict
from pathlib import Path
import random
import logging

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
import torchaudio.transforms as T
from stream_music_gen.constants import inst_name_to_inst_class_id


random.seed(14703)


class Musdb(Dataset):
    """
    Musdb Dataset: https://sigsep.github.io/datasets/musdb.html
    """

    VERSION = "1.0.0"
    URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "original"

    def __init__(
        self,
        root_dir: str = "~/musdb",
        download: bool = True,
        split: str = "train",
        target_sample_rate: int = 16000,
        include_mix: bool = False,
        load_track: bool = False,
        runtime_transform: callable = None,
        regenerate_metadata: bool = False,
    ) -> None:
        """
        Initialize the Musdb dataset.

        Args:
            root_dir (str or Path): Root directory where the dataset is stored or will be downloaded to.
            download (bool): If True, downloads the dataset if it is not already present.
            split (str): The dataset split to use, e.g., 'train', 'test', or 'validation'.
            target_sample_rate (int): The sample rate to which audio files will be resampled.
            include_mix (bool): If True, includes the mixed audio tracks.
            load_track (bool): If True, loads individual tracks.
            runtime_transform (callable, optional): A function/transform to apply to the audio data at runtime.
        """

        self.root_dir = (
            Path(root_dir) if isinstance(root_dir, str) else root_dir
        )
        self.download = download
        self.split = split
        # If target_sample_rate is -1, use the default sample rate
        self.target_sample_rate = (
            target_sample_rate if target_sample_rate > 0 else self.SAMPLE_RATE
        )
        self.runtime_transform = runtime_transform
        self.include_mix = include_mix
        self.load_track = load_track
        self.base_dir = self.root_dir / self.ORIGINAL_DIR_NAME

        self.resample_transform = T.Resample(
            self.SAMPLE_RATE, self.target_sample_rate
        )

        if self.split not in ["train", "test"]:
            raise ValueError("`split` must be one of ['train', 'test'].")

        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()
        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please use `download=True` to download it."
            )
        logging.info(
            f"Found original dataset split {self.split} at {(self.base_dir / self.split).expanduser()}."
        )

        self.all_metadata = self._load_metadata(regenerate_metadata)
        if self.load_track:
            self.all_metadata = (
                self.all_metadata.groupby("track_name").agg(list).reset_index()
            )

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.base_dir / self.split).expanduser()
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        download_and_extract_archive(
            self.URL,
            self.base_dir,
            remove_finished=True,
        )

    def _load_metadata(self, regenerate_metadata: bool = False) -> pd.DataFrame:
        original_dir = (self.base_dir / self.split).expanduser()

        metadata_path = (
            original_dir / f"pt_dataset_metadata_{self.split}.parquet"
        )
        if metadata_path.exists() and not regenerate_metadata:
            return pd.read_parquet(metadata_path)

        tracks = original_dir.glob("*/")
        all_metadata = []
        for track in tqdm(tracks, desc="Loading tracks"):
            stems_paths = [
                path
                for path in track.glob("*.wav")
                if path.name != "mixture.wav"
            ]
            for stem in stems_paths:
                track_name = track.name
                inst_name = stem.stem
                all_metadata.append(
                    {
                        "track_name": track_name,
                        "audio_path": str(stem.relative_to(self.base_dir)),
                        "type": "stem",
                        "instrument_name": inst_name,
                        "program_num": None,
                        "instrument_class_id": inst_name_to_inst_class_id(
                            inst_name
                        ),
                        "original_sample_rate": self.SAMPLE_RATE,
                        "audio_format": "wav",
                    }
                )

            if self.include_mix:
                mix_path = track / "mixture.wav"
                all_metadata.append(
                    {
                        "track_name": track_name,
                        "audio_path": str(mix_path.relative_to(self.base_dir)),
                        "type": "mix",
                        "instrument_name": "mix",
                        "program_num": None,
                        "instrument_class_id": None,
                        "original_sample_rate": self.SAMPLE_RATE,
                        "audio_format": "wav",
                    }
                )

        all_metadata = pd.DataFrame(all_metadata)
        if metadata_path.exists():
            os.remove(metadata_path)
        print(f"Saved metadata to {metadata_path}")
        return all_metadata

    def __len__(self) -> int:
        return len(self.all_metadata)

    def _get_item_audio(self, idx):
        row = self.all_metadata.iloc[idx]
        file_path = str(self.base_dir / row["audio_path"])
        audio = torchaudio.load(file_path)[0]
        audio = self.resample_transform(audio)
        # moises db has 2 channels, take average
        audio = audio.mean(dim=0, keepdim=True)
        base_path = str(Path(file_path).relative_to(self.base_dir))
        item = {
            "audio": audio,
            "file_path": file_path,
            "base_path": base_path,
        }
        if self.runtime_transform:
            item = self.runtime_transform(item)

        return item

    def _get_item_track(self, idx):
        row = self.all_metadata.iloc[idx]
        if self.runtime_transform:
            item = self.runtime_transform(item)
        else:
            raise RuntimeError("Runtime transform is needed for loading track.")
        return row

    def __getitem__(self, idx) -> Dict[str, torch.Tensor | str]:
        if self.load_track:
            return self._get_item_track(idx)
        else:
            return self._get_item_audio(idx)


if __name__ == "__main__":
    dataset = Musdb(
        root_dir="stream_music_gen_data/musdb",
        download=True,
        split="train",
        target_sample_rate=44100,
        runtime_transform=None,
        regenerate_metadata=True,
    )
    print(len(dataset))
    print(dataset[0])
    print(dataset[0]["audio"].shape)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=32
    )
    total_time = 0
    for batch in tqdm(
        dataloader, desc="Iterating over dataset", total=len(dataset)
    ):
        audio = batch["audio"]
        total_time += audio.shape[-1] / dataset.target_sample_rate
    total_time_h = total_time / 3600
    print(f"Total time: {total_time_h:.2f} hours")
