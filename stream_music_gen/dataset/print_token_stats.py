from typing import List
import torch
from stream_music_gen.dataset.token_dataset import TokenDataset
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    data_base_dir = "stream_music_gen_data/causal_dac_32khz"
    splits = ["train", "valid"]
    frame_rate_hz = 50
    dataset_name_all = ["slakh2100", "cocochorales", "moisesdb"]

    lengths = []
    tracks = []
    for dataset_name in dataset_name_all:
        for split in splits:
            dataset = TokenDataset(
                base_dir=data_base_dir,
                dataset_names=[dataset_name],
                split=split,
                transform=torch.nn.Identity(),
            )
            for i in tqdm(range(len(dataset))):
                item = dataset[i]
                tokens = torch.load(
                    item["audio_token_path"][0], weights_only=True
                )
                length = tokens.shape[-1]
                lengths.append(length)
                tracks.append(len(item["audio_token_path"]))
            track_length_hr = sum(lengths) / frame_rate_hz / 3600
            total_length_hr = (
                np.array(lengths) * np.array(tracks) / frame_rate_hz / 3600
            ).sum()
            print(
                f"{dataset_name}-{split}: mean token length: {(sum(lengths) / len(lengths)):.2f}, "
                f"max token length: {max(lengths)}, min token length: {min(lengths)}, "
                f"max tracks: {max(tracks)}, min tracks: {min(tracks)}, "
                f"mean tracks: {(sum(tracks) / len(tracks)):.2f}, "
                f"total hours: {total_length_hr:.2f}, track hours: {track_length_hr:.2f}"
            )
