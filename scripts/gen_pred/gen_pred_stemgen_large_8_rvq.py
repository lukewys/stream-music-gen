"""Generate model predictions using the encoder-decoder model."""

from lightning import seed_everything

import argbind
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
import os
import json
import soundfile as sf
from functools import partial

# weird bug
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from stream_music_gen.lit_module.stemgen import LitStemGen
from stream_music_gen.utils.inference_utils import load_lit_model
from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.utils.audio_utils import (
    mix_with_generated_stem,
    loudness_normalize_audio,
)

GROUP = __file__
# Binding things only when this file is loaded
bind = partial(argbind.bind, group=GROUP)


def generate_prediction(batch, model, tokenizer, device):
    input_token_stems = batch["input_token_stems"]
    enc_inputs = batch["input_emb"].to(device)
    enc_inputs_mask = batch["input_emb_mask"].to(device)
    targets = batch["target_token"].to(device)
    target_inst_id = torch.tensor(
        batch["target_inst_id"], device=enc_inputs.device
    )

    model_pred_cfg = model.generate(
        enc_inputs,
        target_inst_id,
        mask=enc_inputs_mask,
        guidance_scale=2.0,
        confidence_method="vampnet",
        generate_kwargs={
            "sample_temperature": 1.0,
            "noise_temperature": [8.0, 8.0, 4.0, 4.0, 2.0, 2.0, 1.0, 1.0],
            "use_temperature_annealing": True,
        },
        display_pbar=True,
    )

    input_audio_all = []
    target_audio_all = []
    pred_audio_all = []
    num_input_stems_all = []
    for i in range(model_pred_cfg.size(0)):
        # Remove bos and eos tokens and reshape to [num_rvq_layers, frames]
        num_input_stems_all.append(len(batch["input_inst_id"][i]))

        input_tokens = torch.stack(input_token_stems[i], dim=0).to(
            enc_inputs.device
        )
        # Removed summing stems - assume input audio is one mix.
        input_audio_single = (
            tokenizer.codec_tokens_to_audio(
                input_token_stems[i][0].unsqueeze(0)
            )
            .cpu()
            .squeeze()
        )
        input_audio_all.append(input_audio_single)

        target_audio_tokens = tokenizer.post_process_tokens(
            targets[i],
        )
        target_audio = tokenizer.tokens_to_audio(
            target_audio_tokens.unsqueeze(0)
        ).cpu()
        target_audio_all.append(target_audio)

        try:
            pred_audio_tokens_cfg = tokenizer.post_process_tokens(
                model_pred_cfg[i],
            )
            pred_audio_single_cfg = tokenizer.tokens_to_audio(
                pred_audio_tokens_cfg.unsqueeze(0)
            ).cpu()
            pred_audio_all.append(pred_audio_single_cfg)
        except Exception as e:
            print(f"Cannot generate prediction audio: {e}")

    return (
        input_audio_all,
        target_audio_all,
        pred_audio_all,
        num_input_stems_all,
    )


def generate_metadata(batch, idx):
    return {
        "input_inst_id": batch["input_inst_id"][idx],
        "target_inst_id": batch["target_inst_id"][idx],
        "input_token_path": batch["input_token_path"][idx],
        "target_token_path": batch["target_token_path"][idx],
        "input_audio_path": batch["input_audio_path"][idx],
        "target_audio_path": batch["target_audio_path"][idx],
    }


def save_audio_for_one_system(
    input_audio,
    target_or_pred_audio,
    num_input_stems,
    sample_rate,
    sample_dir,
    system_name,
):
    system_dir = sample_dir / system_name
    os.makedirs(system_dir, exist_ok=True)
    target_or_pred_audio_path = system_dir / "pred.wav"
    if not os.path.exists(target_or_pred_audio_path):
        sf.write(target_or_pred_audio_path, target_or_pred_audio, sample_rate)

    # mix
    mix = mix_with_generated_stem(
        input_audio, target_or_pred_audio, num_input_stems, sample_rate
    )
    mix_path = system_dir / "mix.wav"
    if not os.path.exists(mix_path):
        sf.write(mix_path, mix, sample_rate)


@bind
def main(
    save_dir: str = "online-stem-gen/model_predictions/model_predictions_stemgen_val_2025_03_14_large_8_rvq",
    model_path: str = "online-stem-gen/logs/stemgen_large_8_rvq/step=200000.ckpt",
    audio_base_dir: str = "stream_music_gen_data/",
    data_base_dir: str = "stream_music_gen_data/causal_dac_codes_32khz",
    rms_base_dir: str = "stream_music_gen_data/rms_50hz",
    batch_size: int = 16,
    num_samples: int = 1000,
    model_name: str = "stemgen_large_8_rvq",
    gen_kwargs: dict = {},
    seed: Optional[int] = None,
    split: str = "valid",
):
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        seed_everything(seed, workers=True)

    # Load the model
    model, tokenizer, dataloaders = load_lit_model(
        model_path,
        lit_module_cls=LitStemGen,
        batch_size=batch_size,
        override_args={
            "data_base_dir": data_base_dir,
            "rms_base_dir": rms_base_dir,
            "audio_base_dir": audio_base_dir,
        },
    )
    model.eval()
    model.to(device)

    tokenizer.to(device)
    sample_rate = tokenizer.sample_rate

    val_dataloader = get_precomputed_token_dataloader(
        batch_size=batch_size,
        dataset_names=model.config["dataset_names"],
        data_base_dir=model.config["data_base_dir"],
        rms_base_dir=model.config["rms_base_dir"],
        weights=None,
        duration=model.config["max_duration"],
        num_rvq_layers=tokenizer.num_rvq_layers,
        split=split,
        num_workers=8,
        audio_base_dir=audio_base_dir,
        load_audio="true",
        pattern="stemgen",
        shuffle=False,
    )

    file_id = 0

    while True:
        for batch in tqdm(val_dataloader):
            (
                input_audio_all,
                target_audio_all,
                pred_audio_all,
                num_input_stems_all,
            ) = generate_prediction(batch, model, tokenizer, device)

            for idx in range(len(input_audio_all)):
                input_audio = batch["input_audio"][idx].cpu().numpy()
                target_audio = batch["target_audio"][idx].cpu().numpy()
                pred_audio = pred_audio_all[idx].cpu().numpy()
                num_input_stems = num_input_stems_all[idx]
                target_audio_codec = target_audio_all[idx].cpu().numpy()
                input_audio_codec = input_audio_all[idx].cpu().numpy()

                input_audio = loudness_normalize_audio(
                    input_audio, tokenizer.sample_rate
                )
                target_audio = loudness_normalize_audio(
                    target_audio, tokenizer.sample_rate
                )
                pred_audio = loudness_normalize_audio(
                    pred_audio, tokenizer.sample_rate
                )
                target_audio_codec = loudness_normalize_audio(
                    target_audio_codec, tokenizer.sample_rate
                )
                input_audio_codec = loudness_normalize_audio(
                    input_audio_codec, tokenizer.sample_rate
                )

                metadata = generate_metadata(batch, idx)
                metadata["num_input_stems"] = num_input_stems

                sample_dir = save_dir / f"{file_id:05d}"
                os.makedirs(sample_dir, exist_ok=True)

                # save input audio
                input_audio_path = sample_dir / "input_audio.wav"
                if not os.path.exists(input_audio_path):
                    sf.write(input_audio_path, input_audio, sample_rate)
                input_audio_codec_path = sample_dir / "input_audio_codec.wav"
                if not os.path.exists(input_audio_codec_path):
                    sf.write(
                        input_audio_codec_path, input_audio_codec, sample_rate
                    )

                # save target audio
                save_audio_for_one_system(
                    input_audio,
                    target_audio,
                    num_input_stems,
                    sample_rate,
                    sample_dir,
                    "ground_truth",
                )

                save_audio_for_one_system(
                    input_audio,
                    target_audio_codec,
                    num_input_stems,
                    sample_rate,
                    sample_dir,
                    "ground_truth_codec",
                )

                save_audio_for_one_system(
                    input_audio,
                    pred_audio,
                    num_input_stems,
                    sample_rate,
                    sample_dir,
                    model_name,
                )

                # save metadata
                metadata_path = sample_dir / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)

                file_id += 1

            if num_samples > 0 and file_id >= num_samples:
                break

        if num_samples > 0 and file_id >= num_samples:
            break


if __name__ == "__main__":
    args = argbind.parse_args(group=GROUP)
    with argbind.scope(args):
        main()
