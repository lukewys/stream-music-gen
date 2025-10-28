"""Extract RMS features from the dataset classes."""

import os
import argparse

import torch
from pathlib import Path
from tqdm import tqdm
from pqdm.processes import pqdm
import librosa
import numpy as np
import torchaudio

from stream_music_gen.dataset.cocochorales import CocoChorales
from stream_music_gen.dataset.moisesdb import Moisesdb
from stream_music_gen.dataset.musdb import Musdb
from stream_music_gen.dataset.slakh2100 import Slakh2100
from stream_music_gen.constants import DATASET_SPLITS


FRAME_RATE = 50

# Dataset configuration: each dataset is bound with its class
DATASET_CONFIG = {
    "cocochorales": {
        "class": CocoChorales,
    },
    "moisesdb": {
        "class": Moisesdb,
    },
    "musdb": {
        "class": Musdb,
    },
    "slakh2100": {
        "class": Slakh2100,
    },
}


def extract_rms(audio, n_fft, hop_length, sr):
    # audio: (T)
    # Apply A-weighting
    S, phase = librosa.magphase(
        librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    )
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # Replace zero frequency with a small positive value to avoid log10(0)
    frequencies[frequencies == 0] = 1e-10
    A_weighting = librosa.A_weighting(frequencies)
    S_A_weighted = S * np.power(
        10.0, A_weighting[:, None] / 20.0
    )  # Convert A-weighting to linear scale and apply
    rms = librosa.feature.rms(
        S=S_A_weighted, frame_length=n_fft, hop_length=hop_length
    )
    # db scale rms
    rms = librosa.amplitude_to_db(rms).squeeze()
    return rms


def extract_and_save_rms(row, base_dir, output_dir, dataset_name, sample_rate):
    """Extract and save RMS features for a single audio file.

    Args:
        row: Metadata row containing audio_path
        base_dir: Base directory of the dataset
        output_dir: Output directory for RMS features
        dataset_name: Name of the dataset
        sample_rate: Sample rate of the audio
    """
    file_path = str(base_dir / row["audio_path"])
    audio = torchaudio.load(file_path)[0]
    if audio.dim() == 2:
        audio = audio.mean(dim=0, keepdim=True)
    base_path = str(Path(file_path).relative_to(base_dir))
    audio = audio.squeeze().numpy()
    output_path = Path(output_dir) / dataset_name / base_path
    output_path = output_path.with_suffix(".pt")
    if output_path.exists():
        return
    os.makedirs(output_path.parent, exist_ok=True)
    hop_length = int(sample_rate // FRAME_RATE)
    n_fft = hop_length * 2
    rms = extract_rms(audio, n_fft, hop_length, sample_rate)
    rms = torch.tensor(rms)
    torch.save(rms, output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract RMS features from audio datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        default=["all"],
        help="Dataset names list, or use 'all' to process all datasets",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="stream_music_gen_data",
        help="Root directory path for datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stream_music_gen_data/rms_50hz",
        help="Output directory path for RMS features",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--use_original_sample_rate",
        action="store_true",
        help="Use original sample rate from dataset (default: True)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine which datasets to process
    if "all" in args.datasets:
        datasets_to_process = list(DATASET_CONFIG.keys())
    else:
        datasets_to_process = args.datasets

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset_name in datasets_to_process:
        dataset_config = DATASET_CONFIG[dataset_name]
        dataset_class = dataset_config["class"]
        splits = DATASET_SPLITS[dataset_name].values()
        dataset_dir = Path(args.dataset_root) / dataset_name

        for split in splits:
            print(f"[info] Processing {dataset_name}, split: {split}")

            dataset = dataset_class(
                root_dir=str(dataset_dir),
                split=split,
                target_sample_rate=-1,  # use original sample rate
                download=True,
                regenerate_metadata=True,
            )

            base_dir = dataset.base_dir
            sample_rate = dataset.SAMPLE_RATE

            os.makedirs(Path(args.output_dir) / dataset_name, exist_ok=True)
            len_dataset = len(dataset)

            # Prepare arguments for parallel processing
            args_all = []
            for idx in range(len_dataset):
                row = dataset.all_metadata.iloc[idx]
                args_all.append(
                    (row, base_dir, args.output_dir, dataset_name, sample_rate)
                )

            # Process files in parallel
            print(
                f"[info] Extracting RMS for {len_dataset} files with {args.num_workers} workers"
            )
            results = pqdm(
                args_all,
                extract_and_save_rms,
                n_jobs=args.num_workers,
                desc=f"Extracting RMS for {dataset_name}, {split}",
                argument_type="args",
            )

            print(f"[info] Completed {dataset_name}, {split}")
