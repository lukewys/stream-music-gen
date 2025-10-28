"""Extract Encodec codes from the dataset classes."""

import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from stream_music_gen.dataset.cocochorales import CocoChorales
from stream_music_gen.dataset.moisesdb import Moisesdb
from stream_music_gen.dataset.musdb import Musdb
from stream_music_gen.dataset.slakh2100 import Slakh2100
from stream_music_gen.constants import DATASET_SPLITS

from transformers import EncodecModel, AutoProcessor


SAMPLE_RATE = 32000

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


def create_metadata(metadata, base_dir, output_dir):
    metadata = metadata.copy()
    metadata["audio_token_path"] = metadata["audio_path"].apply(
        lambda x: str(
            Path(x.replace(str(base_dir), str(output_dir))).with_suffix(".pt")
        )
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Encodec codes from audio datasets"
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
        default="stream_music_gen_data/encodec_codes_32khz",
        help="Output directory path",
    )
    parser.add_argument(
        "--generate_metadata_only",
        action="store_true",
        help="Only generate metadata without extracting codes",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model + processor (for pre-processing the audio)
    model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
    processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
    bandwidth = model.config.target_bandwidths[-1]  # extract full rvq layers

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
            dataset = dataset_class(
                root_dir=str(dataset_dir),
                split=split,
                target_sample_rate=SAMPLE_RATE,
                download=True,
                regenerate_metadata=True,
            )

            dataset_metadata = dataset.all_metadata
            audio_code_metadata = create_metadata(
                dataset_metadata,
                dataset.base_dir,
                Path(args.output_dir) / dataset_name,
            )

            os.makedirs(Path(args.output_dir) / dataset_name, exist_ok=True)
            metadata_output_path = (
                Path(args.output_dir)
                / dataset_name
                / f"{split}_metadata.parquet"
            )
            if os.path.exists(metadata_output_path):
                os.remove(metadata_output_path)
            audio_code_metadata.to_parquet(metadata_output_path)
            if not args.generate_metadata_only:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers,
                )
                for batch in tqdm(
                    dataloader,
                    desc=f"Extracting Encodec codes for {dataset_name}, {split}",
                    total=len(dataset),
                ):
                    base_path = batch["base_path"][0]
                    output_path = (
                        Path(args.output_dir) / dataset_name / base_path
                    )
                    output_path = output_path.with_suffix(".pt")
                    if output_path.exists():
                        continue
                    audio = batch["audio"]
                    audio = audio.squeeze().numpy()
                    inputs = processor(
                        raw_audio=audio,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        encoder_outputs = model.encode(
                            inputs["input_values"].to(device),
                            inputs["padding_mask"].to(device),
                            bandwidth=bandwidth,
                        )
                        codes = encoder_outputs.audio_codes.squeeze().cpu()
                    os.makedirs(output_path.parent, exist_ok=True)
                    torch.save(codes, output_path)
