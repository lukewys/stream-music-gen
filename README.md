# Codebase for Streaming Generation for Music Accompaniment

[Paper](https://arxiv.org/abs/2510.22105) | [Audio Samples](https://lukewys.github.io/stream-music-gen/)

## Table of Contents

1. [Installation](#installation)
2. [Dataset Dump](#dataset-dump)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Pretrained Checkpoints](#pretrained-checkpoints)

## Installation

First, create an environment with Python >= 3.11 and install [PyTorch](https://pytorch.org/get-started/locally/).

Then, clone and install the codebase:

```bash
git clone https://github.com/lukewys/stream-music-gen.git
cd stream-music-gen
pip install -e .
```

## Dataset Dump

First, you need to download the DAC model weights from [this link](https://huggingface.co/lukewys/stream_music_gen/blob/main/250121_stemmix_dac_weights_400k_steps.pth) and place it in the `pretrained_models/` directory.

Then, extract the audio tokens of tracks in Slakh2100 dataset (the dataset will be automatically downloaded by the codebase):

```bash
python stream_music_gen/dataset/extract_causal_dac_32k.py \
    --datasets slakh2100 \
    --dataset_root stream_music_gen_data \
    --output_dir stream_music_gen_data/causal_dac_codes_32khz \
    --audio_dir stream_music_gen_data \
    --num_workers 8
```

Extract the RMS features of tracks in Slakh2100 dataset for filtering out invalid windows:

```bash
python stream_music_gen/dataset/extract_rms.py \
    --datasets slakh2100 \
    --dataset_root stream_music_gen_data \
    --output_dir stream_music_gen_data/rms_50hz \
    --num_workers 8
```

Use the following command to dump a dataset of sliced and paired examples with the audio mixdown and audio tokens computed in advance.
This is to avoid the need to compute the audio mixdown on the fly during training.

```bash
python stream_music_gen/dataset/dump_audio_mixdown.py \
    --dataset slakh2100 \
    --split train \
    --max_examples 1000000 \
    --output_dir stream_music_gen_data/precompute_audio_mixdown_20s \
    --audio_duration 20 \
    --data_base_dir stream_music_gen_data/causal_dac_codes_32khz \
    --rms_base_dir stream_music_gen_data/rms_50hz \
    --audio_base_dir stream_music_gen_data/

python stream_music_gen/dataset/dump_audio_mixdown.py \
    --dataset slakh2100 \
    --split valid \
    --max_examples 10000 \
    --output_dir stream_music_gen_data/precompute_audio_mixdown_20s \
    --audio_duration 20 \
    --save_audio \
    --data_base_dir stream_music_gen_data/causal_dac_codes_32khz \
    --rms_base_dir stream_music_gen_data/rms_50hz \
    --audio_base_dir stream_music_gen_data/

python stream_music_gen/dataset/dump_audio_mixdown.py \
    --dataset slakh2100 \
    --split test \
    --max_examples 10000 \
    --output_dir stream_music_gen_data/precompute_audio_mixdown_20s \
    --audio_duration 20 \
    --save_audio \
    --data_base_dir stream_music_gen_data/causal_dac_codes_32khz \
    --rms_base_dir stream_music_gen_data/rms_50hz \
    --audio_base_dir stream_music_gen_data/
```

**Notes:**
1. We support the following datasets: `slakh2100`, `cocochorales`, `moisesdb`, and `musdb`. You need to manually download the `musdb` dataset whereas the codebase will automatically download the other datasets.
2. Be aware of your disk space. The dataset roughly takes up 50GB of disk space. The dataset dump has the following sizes: training set: 161GB, validation set: 50GB, test set: 50GB.
3. We dump the audio in 20s to accommodate online models with future visibility > 0 (so it needs to see extra context).

## Training

The training scripts are located in the `scripts/` directory, and the config files are located in the `configs/` directory. Train each type of model by running the command similar to the following:

```bash
# StemGen
python scripts/train_stemgen.py --args.load configs/stemgen/stemgen_large.yml --save_dir logs/stemgen_large

# Prefix Decoder
python scripts/train_prefix_dec.py --args.load configs/prefix_decoder/prefix_decoder_base.yml --save_dir logs/prefix_decoder

# Streaming Model with k=1
python scripts/train_dec_online.py --args.load configs/online/decoder_online_future_visibility_0.yml --save_dir logs/dec_online_future_visibility_0

python scripts/train_dec_online.py --args.load configs/online/decoder_online_future_visibility_-100.yml --save_dir logs/dec_online_future_visibility_-100

# Streaming Model with k>1
python scripts/train_prefix_dec_online.py --args.load configs/online/online_prefix_decoder/online_prefix_decoder_future_visibility_-100_chunk_size_50.yml --save_dir logs/online_prefix_decoder_future_visibility_-100_chunk_size_50

python scripts/train_prefix_dec_online.py --args.load configs/online/online_prefix_decoder/online_prefix_decoder_future_visibility_0_chunk_size_50.yml --save_dir logs/online_prefix_decoder_future_visibility_0_chunk_size_50
```

You can also create a custom config file by copying one of the existing config files and modifying it. This codebase uses [wandb](https://wandb.ai/) to track the training progress. You should create a wandb account and login to the command line to view the training progress.

## Evaluation

### Downloading Pretrained Models for Evaluation
You need to download the pretrained models for evaluation.

We use the [COCOLA](https://github.com/gladia-research-group/cocola) model for accompaniment quality evaluation.
Download the COCOLA checkpoint from [this link](https://drive.google.com/file/d/1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ/view) and place it in the `cocola_models/` directory.

For beat alignment evaluation, we use [Beat This](https://github.com/CPJKU/beat_this), which will automatically download the model weights. We also support [Beat Transformer](https://github.com/zhaojw1998/Beat-Transformer) for beat alignment evaluation. If you use Beat Transformer, download the checkpoint from [this link](https://github.com/zhaojw1998/Beat-Transformer/blob/main/checkpoint/fold_4_trf_param.pt) and place it in the `beat_transformer_models/` directory.

### Running the Evaluation Script

The `scripts/gen_pred/gen_and_evaluate.py` script is used to generate audio examples and evaluate them using the provided model checkpoint.

There is an option to skip the audio generation step if you have already generated the files.

To run this script, you must specify:
    - The model type `stemgen`, `stemgen_large_8_rvq`, `dec_online`, `prefix_decoder`, or `prefix_decoder_online` and `random_anchor`.
    - The path to the model checkpoint, i.e. the `.ckpt` file.

You can optionally pass `batch_size`, `num_samples` to change the batch size used for generation, `data_base_dir` to override the value stored in the checkpoint config, and a few flags to control which evaluation metrics are run.

The `--skip_audio_generation` flag skips the audio generation step and automatically jumps to the evaluation step. Additional flags include `--skip_resampling`, `--skip_beat_alignment`, `--skip_cocola`, and `--skip_fad` to disable specific stages. The evaluation metrics run on 16 kHz audio obtained via resampling, which can be bypassed by specifying `--skip_resampling`, in the case that the resampled files already exist. The `--sub_fad` flag computes FAD between the predicted mixes and ground truth mixes instead of the stems.

### Example Command

```bash
python scripts/gen_pred/gen_and_evaluate.py dec_online /path/to/dec_online_future_visibility_100.ckpt --num_samples 1024
```

### Recommendations

Use 1024 for the number of samples to compute FAD/COCOLA.

## Pretrained Checkpoints

Coming soon!

## A Note on argbind

We use [argbind](https://github.com/pseeth/argbind) to manage config and override arguments of function and class. Argbind only supports kwargs, and does not support bool type and kwargs with None as default value.

We found a bug in `argbind`: if a `bool` keyword argument (`kwarg`) is not loaded from the config, its default value will always be `False`, regardless of what the default value is in the function declaration. However, if the `kwarg` is explicitly specified in the YAML config, it will follow the value set in the config.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{wu2025streaming,
  title         = {Streaming Generation for Music Accompaniment},
  author        = {Wu, Yusong and Wang, Mason and Lei, Heidi and Brade, Stephen and Blanchard, Lancelot and Wu, Shih-Lun and Courville, Aaron and Huang, Anna},
  year          = {2025},
  journal       = {arXiv preprint arXiv:2510.22105},
}
```