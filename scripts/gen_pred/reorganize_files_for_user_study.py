import os
import shutil
from tqdm import tqdm
import glob
import json
import argparse

"""
Rearranges the files in stream_music_gen_models to be more suitable for model-
to model comparisons. Turns the system/example structure into an example/system
structure.
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reorganize Files for User Study."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        help="directory to read generations from",
        default="models/stream_music_gen_models",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="target directory to save reorganized structure",
        default="models/reorganized_examples",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        help="number of examples to reorganize",
        default=1024,
    )
    parser.add_argument(
        "--max_num_stems",
        type=int,
        help="number of examples to reorganize",
        default=-1,
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help=(
            "Point to the split of the dataset, e.g. "
            "precompute_audio_mixdown_20s/slakh2100/valid; used to determine the number of stems"
        ),
        default="stream_music_gen_data/precompute_audio_mixdown_20s/slakh2100/valid",
    )
    args = parser.parse_args()

    # Make target folder
    os.makedirs(args.target_dir, exist_ok=True)

    # Get list of systems
    systems = os.listdir(args.source_dir)

    for i in tqdm(range(args.num_examples)):
        if args.max_num_stems > -1:

            metadata_path = os.path.join(
                args.dataset_dir, f"{i:07d}", "metadata.json"
            )

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            num_stems = len(metadata["input_inst_id"])
            if num_stems > args.max_num_stems:
                continue

        # Make folder for example
        target_example_dir = os.path.join(args.target_dir, f"{i:05d}")
        os.makedirs(target_example_dir, exist_ok=True)

        for system in systems:

            # Make folder for each system:
            # <target_path>/<example>/<system>
            target_system_dir = os.path.join(target_example_dir, system)
            os.makedirs(target_system_dir, exist_ok=True)

            # This is where we are copying from
            source_system_example_dir = os.path.join(
                args.source_dir,
                system,
                f"{i:05d}",
            )

            # Copy predicted stem and mix over
            shutil.copy(
                src=os.path.join(source_system_example_dir, "pred/pred.wav"),
                dst=os.path.join(target_system_dir, "pred.wav"),
            )

            shutil.copy(
                src=os.path.join(source_system_example_dir, "pred/mix.wav"),
                dst=os.path.join(target_system_dir, "mix.wav"),
            )

        # Copy Input audio - input audio is the same, copy from
        # Most recent system
        shutil.copy(
            src=os.path.join(source_system_example_dir, "input_audio.wav"),
            dst=os.path.join(target_example_dir, "input_audio.wav"),
        )

        # Copy Ground Truth
        target_gt_dir = os.path.join(target_example_dir, "ground_truth")
        os.makedirs(target_gt_dir, exist_ok=True)

        shutil.copy(
            src=os.path.join(source_system_example_dir, "ground_truth/mix.wav"),
            dst=os.path.join(target_gt_dir, "mix.wav"),
        )
        shutil.copy(
            src=os.path.join(
                source_system_example_dir, "ground_truth/pred.wav"
            ),
            dst=os.path.join(target_gt_dir, "pred.wav"),
        )
