"""Moisesdb Contrastive Torch Dataset.

Ported from COCOLA: https://github.com/gladia-research-group/cocola
"""

import os
from typing import Dict, Literal
from pathlib import Path
import random
import logging
import json

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

from stream_music_gen.constants import inst_name_to_inst_class_id

random.seed(14703)

# Manually curated split ids for the test split
# 200 tracks from the original dataset are used for the test split
TEST_SPLIT_IDS = [
    "bb6d72a2-63ba-405e-a2c1-c7442b4190bb",
    "2b4e8304-c92d-4347-b09e-cfb9b3e29bf2",
    "fcd1937f-2b21-4a78-889a-7b7e63e0ebdd",
    "11845abc-8ca3-4fb2-bd84-521aeeff56f4",
    "05e7af85-9721-4b42-952a-ccd34feb6033",
    "9efa702f-c6b1-40bb-93ef-095b34a5b775",
    "58efddfb-04b3-4951-858e-e7dbcfccfc21",
    "bbf40b5a-8ef9-4aec-a6c3-8b8706eb2ba0",
    "bdd109ec-d5dd-4d91-92ad-66b679518026",
    "1ade22ad-99bc-4954-b2b9-fb9e8fa06d41",
    "22d265ef-ee2b-4aba-8d60-c3430295cd6d",
    "5640831d-7853-4d06-8166-988e2844b652",
    "b8a79d39-346e-4258-a810-572b3b2c9ab1",
    "c6d73235-1dd5-4085-a3b3-50a3466c6168",
    "4cbd6c36-87a2-4d50-86e3-52d39b98fad3",
    "6a67c964-4514-4bdd-86d4-e290e67ab593",
    "53808b95-cfe9-461d-a113-ffadf32817a1",
    "1921a83e-0373-4bf7-8dc9-6cc9401c9309",
    "444fcc6d-1ad1-4584-b3e5-bf3f4f613c8f",
    "f40ffd10-4e8b-41e6-bd8a-971929ca9138",
]


class Moisesdb(Dataset):
    """
    Moisesdb Dataset: https://github.com/moises-ai/moises-db
    Download Instructions: download the dataset from https://music.ai/research/
    """

    VERSION = "1.0.0"
    SAMPLE_RATE = 44100
    ORIGINAL_DIR_NAME = "moisesdb_v0.1"
    PREPROCESSED_DIR_NAME = "preprocessed"
    PREPROCESSING_INFO_FILE_NAME = "preprocessing_info.json"

    def __init__(
        self,
        root_dir="~/moisesdb",
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

        if download:
            print(
                "Moisesdb dataset need to be downloaded manually "
                "from https://music.ai/research/."
            )

        if not self._is_downloaded_and_extracted():
            raise RuntimeError(f"Dataset split {self.split} not found.")
        logging.info(
            f"Found original dataset split {self.split} at {(self.base_dir).expanduser()}."
        )

        self.all_metadata = self._load_metadata(regenerate_metadata)
        if self.load_track:
            self.all_metadata = (
                self.all_metadata.groupby("track_name").agg(list).reset_index()
            )

    def _is_downloaded_and_extracted(self) -> bool:
        split_dir = (self.base_dir).expanduser()
        return split_dir.exists()

    def _load_metadata(self, regenerate_metadata: bool = False) -> pd.DataFrame:
        original_dir = (self.base_dir).expanduser()

        metadata_path = (
            original_dir / f"pt_dataset_metadata_{self.split}.parquet"
        )
        if metadata_path.exists() and not regenerate_metadata:
            return pd.read_parquet(metadata_path)

        tracks = original_dir.glob("*/")
        all_metadata = []
        for track in tqdm(tracks, desc="Loading tracks"):
            stems_paths = list(track.glob("*/*.wav"))
            metadata = json.load(open(track / "data.json"))
            track_id = track.stem
            stem_metadata_all = {}
            for stem in metadata["stems"]:
                for stem_track in stem["tracks"]:
                    stem_metadata_all[stem_track["id"]] = stem_track

            for stem in stems_paths:
                stem_id = stem.stem
                stem_metadata = stem_metadata_all[stem_id]
                inst_name = stem_metadata["trackType"]
                all_metadata.append(
                    {
                        "track_name": track_id,
                        "audio_path": str(stem.relative_to(self.base_dir)),
                        "type": "stem",
                        "instrument_name": inst_name,
                        "instrument_class_id": inst_name_to_inst_class_id(
                            inst_name
                        ),
                        "original_sample_rate": self.SAMPLE_RATE,
                        "audio_format": "wav",
                    }
                )
            # No mix in Moisesdb

        all_metadata = pd.DataFrame(all_metadata)
        if self.split == "test":
            all_metadata = all_metadata[
                all_metadata["track_name"].isin(TEST_SPLIT_IDS)
            ]
        else:
            all_metadata = all_metadata[
                ~all_metadata["track_name"].isin(TEST_SPLIT_IDS)
            ]
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
    dataset = Moisesdb(
        root_dir="stream_music_gen_data/moisesdb",
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
