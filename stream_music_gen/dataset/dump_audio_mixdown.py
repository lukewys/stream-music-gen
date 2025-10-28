import argparse
import json
import os
from pathlib import Path

from audiotools import AudioSignal
from dac import DAC
import torch
from tqdm import tqdm

from stream_music_gen.constants import (
    DAC_SAMPLE_RATE,
    DAC_PRETRAINED_MODEL_PATH,
)
from stream_music_gen.dataset.token_dataset import get_audio_dataloader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10240, rlimit[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cocochorales", "moisesdb", "musdb", "slakh2100"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid", "test"],
        help="Split name",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        required=True,
        help="Maximum number of examples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="./audio_mixdown",
        help="Output directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers",
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save audio",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index",
    )
    parser.add_argument(
        "--audio_duration",
        type=int,
        default=10,
        help="Audio Duration",
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="stream_music_gen_data/causal_dac_codes_32khz",
        help="Base directory containing the token data",
    )
    parser.add_argument(
        "--rms_base_dir",
        type=str,
        default="stream_music_gen_data/rms_50hz",
        help="Base directory containing RMS data",
    )
    parser.add_argument(
        "--audio_base_dir",
        type=str,
        default="stream_music_gen_data/",
        help="Base directory containing audio files",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = DAC.load(DAC_PRETRAINED_MODEL_PATH).to(device)
    assert model.causal_decoder and model.causal_encoder
    model.eval()
    print(f"[info] Loaded DAC model from {DAC_PRETRAINED_MODEL_PATH}")

    batch_size = 64
    num_example = args.start_index
    pbar = tqdm(
        total=args.max_examples,
        desc=f"Extracting DAC codes for {args.dataset}, {args.split}",
        initial=args.start_index,
    )
    dataloader = get_audio_dataloader(
        batch_size=batch_size,
        num_workers=args.num_workers,
        split=args.split,
        dataset_names=[args.dataset],
        weights=[1],
        group_by_track="true",
        filter_stem_by_rms="true",
        data_base_dir=args.data_base_dir,
        rms_base_dir=args.rms_base_dir,
        audio_base_dir=args.audio_base_dir,
        target_sample_rate=DAC_SAMPLE_RATE,
        duration=args.audio_duration,
    )

    while True:
        for batch in dataloader:
            for i in range(len(batch["input_audio"])):
                output_path = (
                    Path(args.output_dir)
                    / args.dataset
                    / args.split
                    / f"{num_example:07d}"
                )
                os.makedirs(output_path, exist_ok=True)
                for audio_type in ["input", "target"]:
                    audio = AudioSignal(
                        batch[f"{audio_type}_audio"][i],
                        sample_rate=DAC_SAMPLE_RATE,
                    ).to(device)

                    # NOTE(Shih-Lun): ensure no chunking
                    win_duration = audio.shape[-1] / DAC_SAMPLE_RATE + 1

                    with torch.no_grad():
                        encoder_outputs = model.compress(
                            audio, win_duration=win_duration
                        )
                        codes = encoder_outputs.codes.squeeze().cpu()

                    torch.save(codes, output_path / f"{audio_type}_codes.pt")
                    if args.save_audio:
                        torch.save(
                            audio, output_path / f"{audio_type}_audio.pt"
                        )

                # select only the metadata we need
                metadata = {
                    k: batch[k][i]
                    for k in [
                        "base_dir",
                        "input_inst_id",
                        "target_inst_id",
                        "input_token_path",
                        "target_token_path",
                        "input_audio_path",
                        "target_audio_path",
                    ]
                }
                # add num_stems to metadata
                metadata["num_stems"] = len(batch["input_inst_id"][i])

                with open(
                    output_path / "metadata.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(metadata, f)

                num_example += 1
                pbar.update(1)
                if num_example >= args.max_examples:
                    return


if __name__ == "__main__":
    main()
