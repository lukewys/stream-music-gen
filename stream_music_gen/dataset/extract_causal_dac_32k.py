"""Extract (causal) DAC codes from the dataset classes."""

import os
import argparse

import torch
from pathlib import Path
from tqdm import tqdm
from audiotools import AudioSignal
from dac import DAC

from stream_music_gen.dataset.cocochorales import CocoChorales
from stream_music_gen.dataset.moisesdb import Moisesdb
from stream_music_gen.dataset.musdb import Musdb
from stream_music_gen.dataset.slakh2100 import Slakh2100
from stream_music_gen.constants import DAC_PRETRAINED_MODEL_PATH, DATASET_SPLITS


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


def create_metadata(metadata, base_dir, output_dir, audio_dir):
    metadata = metadata.copy()
    metadata["audio_token_path"] = metadata["audio_path"].apply(
        lambda x: str(
            Path(x.replace(str(base_dir), str(output_dir))).with_suffix(".pt")
        )
    )
    metadata["audio_path"] = metadata["audio_path"].apply(
        lambda x: str((base_dir / Path(x)).relative_to(audio_dir))
    )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract (causal) DAC codes from audio datasets"
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
        default="stream_music_gen_data/causal_dac_codes_32khz",
        help="Output directory path",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="stream_music_gen_data",
        help="Audio directory path for metadata generation",
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

    # load the model
    model = DAC.load(DAC_PRETRAINED_MODEL_PATH).to(device)
    assert model.causal_decoder and model.causal_encoder
    model.eval()
    print(f"[info] Loaded DAC model from {DAC_PRETRAINED_MODEL_PATH}")

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
                args.audio_dir,
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
                    desc=f"Extracting DAC codes for {dataset_name}, {split}",
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

                    audio = AudioSignal(audio, sample_rate=SAMPLE_RATE).to(
                        device
                    )
                    # NOTE(Shih-Lun): ensure no chunking
                    win_duration = audio.shape[-1] / SAMPLE_RATE + 1

                    with torch.no_grad():
                        encoder_outputs = model.compress(
                            audio, win_duration=win_duration
                        )
                        codes = encoder_outputs.codes.squeeze().cpu()

                    os.makedirs(output_path.parent, exist_ok=True)
                    torch.save(codes, output_path)
                    # print(codes.size())

                    # NOTE(Shih-Lun): decode to sanity check
                    # with torch.no_grad():
                    #     recons = model.decompress(encoder_outputs).cpu()

                    # print(output_path)
                    # recons.write(str(output_path).replace(".pt", "_recons.wav"))
