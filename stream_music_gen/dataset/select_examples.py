"""Select the exact list of examples to precompute (an "example list").

An example is a (track, target stem, ordered input stems, crop start) tuple,
chosen with the same rules as on-the-fly sampling (`SelectStems.select`).
Selection only needs the RMS curves, so this step is fast and CPU-only.
`dump_audio_mixdown.py --example_list` then materializes the examples
without any randomness, so a published example list lets anyone rebuild
exactly the same dataset (e.g. the evaluation sets) and compare results.

Example list format (JSON Lines, gzipped if the path ends with `.gz`):
    line 1:  {"meta": {...}, "stems": [{"track", "name", "audio_path",
              "token_path", "inst_id"}, ...]}
    line 2+: {"idx", "dataset", "track", "target", "inputs", "start_frame"}
`target` and `inputs` index into `stems`; `inputs` keeps the sampled order
(the input mixdown is a mean, so order matters for bit-exact audio);
`start_frame` is the crop start in 50 Hz frames.

Usage:
    python stream_music_gen/dataset/select_examples.py \
        --dataset slakh2100 --split test --num_examples 10000 \
        --output example_lists/slakh2100_test.jsonl.gz \
        --data_base_dir stream_music_gen_data/causal_dac_codes_32khz \
        --rms_base_dir stream_music_gen_data/rms_50hz \
        --audio_base_dir stream_music_gen_data/
"""

import argparse
import gzip
import json
import random
from pathlib import Path

from tqdm import tqdm

from stream_music_gen.dataset.token_dataset import (
    TokenDataset,
    get_audio_select_stems,
)
from stream_music_gen.dataset.transforms import LoadRMS


def write_example_list(path, meta, stems, examples):
    lines = [json.dumps({"meta": meta, "stems": stems})]
    lines += [json.dumps(example) for example in examples]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    if str(path).endswith(".gz"):
        # mtime=0: the same examples always give a byte-identical file.
        data = gzip.compress(data, mtime=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)


def load_example_list(path):
    """Returns (meta, stems, examples)."""
    data = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        data = gzip.decompress(data)
    header, *lines = data.decode("utf-8").splitlines()
    header = json.loads(header)
    examples = [json.loads(line) for line in lines]
    return header["meta"], header["stems"], examples


def select_examples(
    dataset_name: str,
    split: str,
    num_examples: int,
    seed: int,
    data_base_dir: str,
    rms_base_dir: str,
    audio_base_dir: str,
    audio_duration: int,
    max_num_input_stems=None,
):
    """Select `num_examples` examples; returns (meta, stems, examples).

    Like the on-the-fly loader, every "epoch" visits each track once in a
    random order and draws one example from it.
    """
    # SelectStems draws from the global `random` module.
    random.seed(seed)

    token_dataset = TokenDataset(
        base_dir=data_base_dir,
        dataset_names=[dataset_name],
        split=split,
        group_by_track=True,
        transform=LoadRMS(rms_base_dir=rms_base_dir),
    )
    select_stems = get_audio_select_stems(
        audio_base_dir=audio_base_dir,
        duration=audio_duration,
        max_num_input_stems=max_num_input_stems,
    )

    dataset_base_dir = Path(data_base_dir) / dataset_name
    tracks, track_stem_ids, stems = [], [], []
    for i in tqdm(range(len(token_dataset)), desc="Loading RMS"):
        item = token_dataset[i]
        ids = []
        for audio_path, token_path, inst_id in zip(
            item["audio_path"],
            item["audio_token_path"],
            item["instrument_class_id"],
        ):
            ids.append(len(stems))
            stems.append(
                {
                    "track": item["track_name"],
                    "name": Path(audio_path).stem,
                    "audio_path": audio_path,
                    "token_path": str(
                        Path(token_path).relative_to(dataset_base_dir)
                    ),
                    "inst_id": int(inst_id),
                }
            )
        tracks.append(item)
        track_stem_ids.append(ids)

    examples = []
    pbar = tqdm(total=num_examples, desc=f"Selecting {dataset_name}-{split}")
    while len(examples) < num_examples:
        order = list(range(len(tracks)))
        random.shuffle(order)
        for t in order:
            if len(examples) >= num_examples:
                break
            input_idxs, target_idx, start = select_stems.select(tracks[t])
            ids = track_stem_ids[t]
            examples.append(
                {
                    "idx": len(examples),
                    "dataset": dataset_name,
                    "track": tracks[t]["track_name"],
                    "target": ids[target_idx],
                    "inputs": [ids[i] for i in input_idxs],
                    "start_frame": int(start),
                }
            )
            pbar.update(1)
    pbar.close()

    meta = {
        "dataset": dataset_name,
        "split": split,
        "num_examples": num_examples,
        "seed": seed,
        "audio_duration": audio_duration,
        "frame_rate_hz": select_stems.rms_token_hz,
        "selection": {
            "ignore_instrument_ids": select_stems.ignore_instrument_ids,
            "max_num_input_stems": select_stems.max_num_input_stems,
            "token_crop_length": select_stems.token_crop_length,
            "filter_stem_by_rms": select_stems.filter_stem_by_rms,
            "rms_threshold": select_stems.rms_threshold,
            "num_retries": select_stems.num_retries,
            "min_overlap_portion": select_stems.min_overlap_portion,
        },
    }
    return meta, stems, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cocochorales", "moisesdb", "musdb", "slakh2100"],
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "valid", "test"]
    )
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output example list path (.jsonl or .jsonl.gz)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio_duration", type=int, default=20)
    parser.add_argument("--max_num_input_stems", type=int, default=None)
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

    meta, stems, examples = select_examples(
        dataset_name=args.dataset,
        split=args.split,
        num_examples=args.num_examples,
        seed=args.seed,
        data_base_dir=args.data_base_dir,
        rms_base_dir=args.rms_base_dir,
        audio_base_dir=args.audio_base_dir,
        audio_duration=args.audio_duration,
        max_num_input_stems=args.max_num_input_stems,
    )
    write_example_list(args.output, meta, stems, examples)
    print(
        f"[info] Wrote {len(examples)} examples ({len(stems)} stems) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
