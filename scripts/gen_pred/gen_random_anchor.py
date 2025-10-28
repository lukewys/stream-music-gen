from lightning import seed_everything

import argparse
from typing import Optional
import os
import json
import numpy as np
import torch
import soundfile as sf
import random
import glob
from pathlib import Path
from tqdm import tqdm
from stream_music_gen.constants import DAC_SAMPLE_RATE

from stream_music_gen.utils.audio_utils import (
    mix_with_generated_stem,
    loudness_normalize_audio,
)


def save_audio_pair(
    save_dir: Path,
    mix_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int = DAC_SAMPLE_RATE,
) -> None:
    """Save a pair of mixed and target audio files."""
    os.makedirs(save_dir, exist_ok=True)
    sf.write(os.path.join(save_dir, "mix.wav"), mix_audio, sample_rate)
    sf.write(os.path.join(save_dir, "pred.wav"), target_audio, sample_rate)


def generate_metadata_for_random_anchor(original_metadata, random_target_dir):
    """Generate metadata for random anchor, combining original and random target info."""
    return {
        "input_inst_id": original_metadata["input_inst_id"],
        "target_inst_id": original_metadata["target_inst_id"],
        "input_token_path": original_metadata.get("input_token_path", ""),
        "target_token_path": original_metadata.get("target_token_path", ""),
        "input_audio_path": original_metadata.get("input_audio_path", ""),
        "target_audio_path": original_metadata.get("target_audio_path", ""),
        "random_target_dir": str(
            random_target_dir
        ),  # Add info about random target source
        "generation_type": "random_anchor",
    }


def main(
    model_path: str = None,  # Not used for random anchor but needed for compatibility
    save_dir: str = None,
    data_base_dir: str = None,
    batch_size: int = 64,  # Not used but needed for compatibility
    model_name: str = "pred",
    num_samples: int = 1024,
    gen_kwargs: dict = {},  # Not used but needed for compatibility
    split: str = "valid",
    # Legacy parameters for backward compatibility
    duration: float = 10,
    sample_rate: int = 32000,
    seed: Optional[int] = None,
    **kw_args,
):
    if seed is not None:
        seed_everything(seed)
    dataset_dir = os.path.join(data_base_dir, "slakh2100", split)

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    # Create a mapping from target instrument ID to example
    examples_for_each_target_id = {}
    samples_list = sorted([x for x in dataset_dir.glob("*") if x.is_dir()])
    for sample_dir in tqdm(samples_list, desc="Gather Samples"):
        with open(sample_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        target_instrument_id = metadata["target_inst_id"]

        if target_instrument_id in examples_for_each_target_id:
            examples_for_each_target_id[target_instrument_id].append(sample_dir)
        else:
            examples_for_each_target_id[target_instrument_id] = [sample_dir]

    # Shuffle
    for k, v in examples_for_each_target_id.items():
        random.shuffle(v)

    # Main loop - iterate and save audio
    for i, sample_dir in enumerate(tqdm(samples_list, desc="Saving Audios")):

        if i >= num_samples:
            break

        # Load metadata
        with open(sample_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Load input and target:
        input_audio = torch.load(os.path.join(sample_dir, "input_audio.pt"))
        gt_target_audio = torch.load(
            os.path.join(sample_dir, "target_audio.pt")
        )
        # This truncation what the dataloader appears to be doing.
        # Double check with Yusong/Lancelot
        input_audio = (
            input_audio[..., : int(duration * sample_rate)].numpy().reshape(-1)
        )
        gt_target_audio = (
            gt_target_audio[..., : int(duration * sample_rate)]
            .numpy()
            .reshape(-1)
        )

        input_audio = loudness_normalize_audio(input_audio, sample_rate)
        gt_target_audio = loudness_normalize_audio(gt_target_audio, sample_rate)
        gt_mix = mix_with_generated_stem(
            input_audio,
            gt_target_audio,
            num_stems=len(metadata["input_inst_id"]),
            sample_rate=sample_rate,
        )

        # Get intended target instrument
        target_instrument_id = metadata["target_inst_id"]

        # Baseline - retrieve target audio of same instrument, using pop()
        rand_target_audio_dir = examples_for_each_target_id[
            target_instrument_id
        ].pop()
        rand_target_audio = torch.load(
            os.path.join(rand_target_audio_dir, "target_audio.pt")
        )
        rand_target_audio = (
            rand_target_audio[..., : int(duration * sample_rate)]
            .numpy()
            .reshape(-1)
        )
        rand_target_audio = loudness_normalize_audio(
            rand_target_audio, sample_rate
        )
        pred_mix = mix_with_generated_stem(
            input_audio,
            rand_target_audio,
            num_stems=len(metadata["input_inst_id"]),
            sample_rate=sample_rate,
        )

        # Save
        example_dir = os.path.join(save_dir, f"{i:05d}")
        os.makedirs(example_dir, exist_ok=True)
        sf.write(
            os.path.join(example_dir, "input_audio.wav"),
            input_audio,
            sample_rate,
        )
        gt_dir = os.path.join(example_dir, "ground_truth")
        pred_dir = os.path.join(example_dir, model_name)

        save_audio_pair(
            save_dir=gt_dir,
            mix_audio=gt_mix,
            target_audio=gt_target_audio,
            sample_rate=sample_rate,
        )
        save_audio_pair(
            save_dir=pred_dir,
            mix_audio=pred_mix,
            target_audio=rand_target_audio,
            sample_rate=sample_rate,
        )

        # Generate and save metadata
        new_metadata = generate_metadata_for_random_anchor(
            metadata, rand_target_audio_dir
        )
        new_metadata["num_input_stems"] = len(metadata["input_inst_id"])

        metadata_path = os.path.join(example_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(new_metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Random Anchors.")
    parser.add_argument("save_dir", type=str, help="Dir to save examples into")
    parser.add_argument(
        "--model_name",
        type=str,
        default="pred",
        help="Name of the model inside save_dir, usually pred",
    )

    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="",
        help="Base directory for data",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        choices=["valid", "test", "train"],
        help="Dataset split to use",
    )

    parser.add_argument(
        "--num_samples", type=int, default=1024, help="Num. Samples to Generate"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=32000, help="Audio Sample Rate"
    )
    parser.add_argument(
        "--duration", type=float, default=10, help="Duration in seconds"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    main(
        save_dir=args.save_dir,
        model_name=args.model_name,
        num_samples=args.num_samples,
        data_base_dir=args.data_base_dir,
        split=args.split,
        duration=args.duration,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
