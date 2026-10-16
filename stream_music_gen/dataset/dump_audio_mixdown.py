import argparse
import json
import os
from pathlib import Path

from audiotools import AudioSignal
from dac import DAC
from lightning import seed_everything
import torch
from tqdm import tqdm

from stream_music_gen.constants import (
    DAC_SAMPLE_RATE,
    DAC_PRETRAINED_MODEL_PATH,
)
from stream_music_gen.dataset.select_examples import load_example_list
from stream_music_gen.dataset.token_dataset import (
    get_audio_dataloader,
    get_example_list_audio_dataloader,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10240, rlimit[1]))


METADATA_KEYS = [
    "base_dir",
    "input_inst_id",
    "target_inst_id",
    "input_token_path",
    "target_token_path",
    "input_audio_path",
    "target_audio_path",
    "crop_start_frame",
]

# Examples per worker batch in example list mode. Small batches keep all
# workers busy; the DAC encoder below processes one example at a time anyway.
EXAMPLE_LIST_BATCH_SIZE = 8


def encode_and_save(
    batch, i, output_path, model, device, save_audio, extra_metadata=None
):
    """Encode example `i` of `batch` with DAC and write it to `output_path`."""
    os.makedirs(output_path, exist_ok=True)
    for audio_type in ["input", "target"]:
        audio = AudioSignal(
            batch[f"{audio_type}_audio"][i],
            sample_rate=DAC_SAMPLE_RATE,
        ).to(device)

        # NOTE(Shih-Lun): ensure no chunking
        win_duration = audio.shape[-1] / DAC_SAMPLE_RATE + 1

        with torch.no_grad():
            encoder_outputs = model.compress(audio, win_duration=win_duration)
            codes = encoder_outputs.codes.squeeze().cpu()

        torch.save(codes, output_path / f"{audio_type}_codes.pt")
        if save_audio:
            torch.save(audio, output_path / f"{audio_type}_audio.pt")

    # select only the metadata we need
    metadata = {k: batch[k][i] for k in METADATA_KEYS}
    # add num_stems to metadata
    metadata["num_stems"] = len(batch["input_inst_id"][i])
    metadata.update(extra_metadata or {})

    # Written last: its presence marks a complete example (used to resume).
    with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)


def dump_from_example_list(args, model, device):
    """Materialize the examples of an example list (see select_examples.py)."""
    meta, stems, examples = load_example_list(args.example_list)
    split_dir = Path(args.output_dir) / meta["dataset"] / meta["split"]

    shard = [
        ex for ex in examples if ex["idx"] % args.num_shards == args.shard_id
    ]
    todo = [
        ex
        for ex in shard
        if not (split_dir / f"{ex['idx']:07d}" / "metadata.json").exists()
    ]
    print(
        f"[info] Example list {args.example_list}: {len(examples)} examples, "
        f"shard {args.shard_id}/{args.num_shards} has {len(shard)}, "
        f"{len(shard) - len(todo)} already done."
    )
    if not todo:
        return

    dataloader = get_example_list_audio_dataloader(
        examples=todo,
        stems=stems,
        data_base_dir=args.data_base_dir,
        audio_base_dir=args.audio_base_dir,
        duration=meta["audio_duration"],
        target_sample_rate=DAC_SAMPLE_RATE,
        batch_size=EXAMPLE_LIST_BATCH_SIZE,
        num_workers=args.num_workers,
    )
    pbar = tqdm(
        total=len(todo),
        desc=f"Extracting DAC codes for {meta['dataset']}, {meta['split']}",
    )
    for batch in dataloader:
        for i in range(len(batch["input_audio"])):
            idx = batch["example_idx"][i]
            encode_and_save(
                batch,
                i,
                split_dir / f"{idx:07d}",
                model,
                device,
                args.save_audio,
                extra_metadata={"example_idx": idx},
            )
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_list",
        type=str,
        default=None,
        help="Dump exactly the examples listed in this example list (see "
        "select_examples.py) instead of sampling them on the fly",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="With --example_list: only dump examples with "
        "idx %% num_shards == shard_id",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="With --example_list: number of shards",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["cocochorales", "moisesdb", "musdb", "slakh2100"],
        help="Dataset name (required without --example_list)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "valid", "test"],
        help="Split name (required without --example_list)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process "
        "(required without --example_list)",
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
        default=None,
        help="Audio Duration (default: 10; ignored with --example_list)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    if args.example_list is None:
        for name in ["dataset", "split", "max_examples"]:
            if getattr(args, name) is None:
                parser.error(f"--{name} is required without --example_list")
        if args.audio_duration is None:
            args.audio_duration = 10

    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    # Timing-based algorithm selection would make DAC codes vary between runs.
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = DAC.load(DAC_PRETRAINED_MODEL_PATH).to(device)
    assert model.causal_decoder and model.causal_encoder
    model.eval()
    print(f"[info] Loaded DAC model from {DAC_PRETRAINED_MODEL_PATH}")

    # The first DAC call in a process returns slightly different codes than
    # all later calls (one-time CUDA initialization), so warm up once.
    warmup = torch.randn(
        DAC_SAMPLE_RATE, generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        model.compress(
            AudioSignal(0.1 * warmup, sample_rate=DAC_SAMPLE_RATE).to(device),
            win_duration=2,
        )

    if args.example_list is not None:
        dump_from_example_list(args, model, device)
        return

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
                encode_and_save(
                    batch, i, output_path, model, device, args.save_audio
                )

                num_example += 1
                pbar.update(1)
                if num_example >= args.max_examples:
                    return


if __name__ == "__main__":
    main()
