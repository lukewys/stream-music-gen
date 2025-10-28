"""CocoChorales Contrastive Torch Dataset."""

import os
from typing import Dict, Literal
from pathlib import Path
import shutil
import random
import logging
import yaml

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
import torchaudio
import torchaudio.transforms as T

from stream_music_gen.constants import midi_prog_to_inst_class_id

random.seed(14703)


class CocoChorales(Dataset):
    """
    CocoChorales Dataset: https://magenta.tensorflow.org/datasets/cocochorales
    """

    VERSION = "1.0.0"
    URLS = {
        "train": [
            f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/train/{i}.tar.bz2"
            for i in [1, 2, 3, 25, 26, 27, 49, 50, 51, 73, 74, 75]
        ],
        "test": [
            f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/test/{i}.tar.bz2"
            for i in [1, 4, 7, 10]
        ],
        "valid": [
            f"https://storage.googleapis.com/magentadata/datasets/cocochorales/cocochorales_full_v1_zipped/main_dataset/valid/{i}.tar.bz2"
            for i in [1, 4, 7, 10]
        ],
    }
    MD5S = {
        "train": [
            "999ba8284b0646a6c7f3ef357e15fd59",
            "f1b6ae484940d66ec006c0599d8b0f48",
            "b2237240c49d3537872d35d98199fdc6",
            "e540dc37fcb47f75995544df3720af3f",
            "7490eb20468f421313bab7882f59c9cf",
            "200eb27e786d27d04347129d10a7731b",
            "358817b12ee126e697f14ef6805cdc48",
            "96e81212eeb8b65619103dd16094a08f",
            "32799d360b9b9764b511d399327509e0",
            "0fa937613c947d0cc18d2d4682504fa0",
            "e5c50a10b0b2af5ee26867c108a94a92",
            "f78dfe2f212e4991a78be7e8e4e98fc5",
        ],
        "test": [
            "2c9e617b9f3ec622e0da35734036af49",
            "461fc00182c5e312ac379d97df4bceb6",
            "f808fc2502059e9a994cea85ccd4d3a0",
            "afebac996cd3d643b7c99d575a3ad048",
        ],
        "valid": [
            "697766f8e53ffc9f64708b8bf4acedb1",
            "4edd6803d082dc090f08823cc003cc94",
            "502128334a38e682ac0a06682207d13b",
            "ded860cabdf005eafde1095ebab7787e",
        ],
    }
    SAMPLE_RATE = 16000
    ORIGINAL_DIR_NAME = "original"
    PREPROCESSED_DIR_NAME = "preprocessed"
    PREPROCESSING_INFO_FILE_NAME = "preprocessing_info.json"

    def __init__(
        self,
        root_dir="~/coco_chorales_contrastive",
        download: bool = True,
        split: str = "train",
        target_sample_rate: int = 16000,
        include_mix: bool = False,
        load_track: bool = False,
        runtime_transform: callable = None,
        regenerate_metadata: bool = False,
    ) -> None:
        """
        Initialize the Slakh2100ContrastivePreprocessed dataset.

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

        if self.split not in ["train", "valid", "test"]:
            raise ValueError(
                "`split` must be one of ['train', 'valid', 'test']."
            )

        if self.download and not self._is_downloaded_and_extracted():
            self._download_and_extract()
        if not self._is_downloaded_and_extracted():
            raise RuntimeError(
                f"Dataset split {self.split} not found. Please use `download=True` to download it."
            )
        logging.info(
            f"Found original dataset split {self.split} at {(self.base_dir / self.split).expanduser()}."
        )

        # delete string_track001353 from train set
        split_dir = (self.base_dir / self.split).expanduser()
        if split_dir / "string_track001353" in split_dir.iterdir():
            print("Deleting string_track001353 from train set")
            shutil.rmtree(split_dir / "string_track001353")

        self.all_metadata = self._load_metadata(regenerate_metadata)
        if self.load_track:
            self.all_metadata = (
                self.all_metadata.groupby("track_name").agg(list).reset_index()
            )

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.base_dir / self.split).expanduser()
        return split_dir.exists() and any(split_dir.iterdir())

    def _download_and_extract(self) -> None:
        for i, url in enumerate(self.URLS[self.split]):
            download_and_extract_archive(
                url,
                self.base_dir / self.split,
                md5=self.MD5S[self.split][i],
                remove_finished=True,
            )

    def _load_metadata(self, regenerate_metadata: bool = False) -> pd.DataFrame:
        original_dir = (self.base_dir / self.split).expanduser()

        metadata_path = (
            original_dir / f"pt_dataset_metadata_{self.split}.parquet"
        )
        if metadata_path.exists() and not regenerate_metadata:
            return pd.read_parquet(metadata_path)

        tracks = list(original_dir.glob("*/"))
        all_metadata = []
        for track in tqdm(tracks, desc="Loading tracks"):
            stems_paths = list(track.glob("stems_audio/*.wav"))
            yaml_text = (track / "metadata.yaml").read_text()
            # if the first character is not a valid yaml character, remove it
            if yaml_text[0] not in ["{", "-"]:
                yaml_text = yaml_text[1:]
            metadata = yaml.safe_load(yaml_text)
            track_id = track.stem
            for stem in stems_paths:
                stem_id = int(stem.stem.split("_")[0]) - 1
                all_metadata.append(
                    {
                        "track_name": track_id,
                        "audio_path": str(stem.relative_to(self.base_dir)),
                        "type": "stem",
                        "instrument_name": metadata["instrument_name"][stem_id],
                        "program_num": metadata["midi_program_number"][stem_id],
                        # midi_program_number is 0-indexed
                        "instrument_class_id": midi_prog_to_inst_class_id(
                            metadata["midi_program_number"][stem_id] + 1
                        ),
                        "original_sample_rate": self.SAMPLE_RATE,
                        "audio_format": "wav",
                    }
                )

            if self.include_mix:
                mix = track / "mix.wav"
                all_metadata.append(
                    {
                        "track_name": track_id,
                        "audio_path": str(mix.relative_to(self.base_dir)),
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
        all_metadata.to_parquet(metadata_path)
        print(f"Saved metadata to {metadata_path}")
        return all_metadata

    def __len__(self) -> int:
        return len(self.all_metadata)

    def _get_item_audio(self, idx):
        row = self.all_metadata.iloc[idx]
        file_path = str(self.base_dir / row["audio_path"])
        audio = torchaudio.load(file_path)[0]
        audio = self.resample_transform(audio)
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
    dataset = CocoChorales(
        root_dir="stream_music_gen_data/cocochorales",
        download=True,
        split="valid",
        target_sample_rate=16000,
        runtime_transform=None,
        regenerate_metadata=True,
    )
    dataset = CocoChorales(
        root_dir="stream_music_gen_data/cocochorales",
        download=True,
        split="test",
        target_sample_rate=16000,
        runtime_transform=None,
        regenerate_metadata=True,
    )
    dataset = CocoChorales(
        root_dir="stream_music_gen_data/cocochorales",
        download=True,
        split="train",
        target_sample_rate=16000,
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
