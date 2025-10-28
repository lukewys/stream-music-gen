# Evaluates Beat Alignment using BeatTransformer's method
import argparse
import os
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

import librosa
from beat_transformer.DilatedTransformer import Demixed_DilatedTransformerModel
from scipy.signal.windows import hann
import torch
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.evaluation.beats import BeatEvaluation
from beat_this.inference import File2Beats

MODEL_PATH = "beat_transformer_models/fold_4_trf_param.pt"

SR = 44100
N = 4096
H = 1024
FPS = SR / H
window = hann(N, sym=False)
mel_f = librosa.filters.mel(sr=SR, n_fft=N, n_mels=128, fmin=30, fmax=11000).T

model = Demixed_DilatedTransformerModel(
    attn_len=5,
    instr=5,
    ntoken=2,
    dmodel=256,
    nhead=8,
    d_hid=H,
    nlayers=9,
    norm_first=True,
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))["state_dict"]
)
model.eval()

beat_tracker = DBNBeatTrackingProcessor(
    min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=10, threshold=0.2
)

device = "cuda" if torch.cuda.is_available() else "cpu"
file2beats = File2Beats(checkpoint_path="final0", device=device, dbn=False)


def preprocess_audio(audio_file):
    audio, _ = librosa.load(audio_file, sr=44100)
    data = np.asfortranarray(audio)
    audio_padded = np.concatenate((np.zeros((N,)), data, np.zeros((N,))))

    stft_res = librosa.stft(
        audio_padded, n_fft=N, hop_length=1024, window=window, center=False
    ).T
    stft_res = np.dot(np.abs(stft_res) ** 2, mel_f).T
    stft_res = librosa.power_to_db(stft_res, ref=np.max).T
    stft_res = np.expand_dims(stft_res, axis=0)

    return stft_res


def beat_alignment_score(
    root_folder: str,
    context_path: str = "input_audio.wav",
    gt_path: str = "ground_truth/pred.wav",
    pred_path: str = "stemgen_base/pred.wav",
    method: str = "beat_this",
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Computes BeatTransformer's alignment scores for all generation folders in
    the root directory.

    Args:
        root_folder (str): Path to the root directory containing folders of
            generation examples.
        context_path (str): Relative path from single example folder to
            the context audio.
        gt_path (str): Relative path from single example folder to
            ground truth audio.
        pred_path (str): Relative path from single example folder
            to predicted audio.
        method (str): Method to use for beat alignment.

    Returns:
        Tuple[Dict[str, list], Dict[str, list]]: Two dictionaries containing all
            scores for ground truth (GT) and predicted (PRED) audio.
    """

    generation_paths = sorted(
        [
            os.path.join(root_folder, d)
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
    )
    all_gt_scores = {}
    all_pred_scores = {}

    for generation_path in tqdm(
        generation_paths, desc="Processing Beat Alignment Scores"
    ):
        if method == "beat_this":
            gt_scores, pred_scores = (
                compute_beat_alignment_given_path_beat_this(
                    generation_path, context_path, gt_path, pred_path
                )
            )
        elif method == "beat_transformer":
            gt_scores, pred_scores = (
                compute_beat_alignment_given_path_beat_transformer(
                    generation_path, context_path, gt_path, pred_path
                )
            )
        else:
            raise ValueError(f"Invalid method: {method}")

        for key, value in gt_scores.items():
            if key not in all_gt_scores:
                all_gt_scores[key] = []
            all_gt_scores[key].append(
                value.item() if isinstance(value, torch.Tensor) else value
            )
        for key, value in pred_scores.items():
            if key not in all_pred_scores:
                all_pred_scores[key] = []
            all_pred_scores[key].append(
                value.item() if isinstance(value, torch.Tensor) else value
            )

    return all_gt_scores, all_pred_scores


def compute_beat_alignment_given_path_beat_transformer(
    generation_path: str,
    context_path: str = "input_audio.wav",
    gt_path: str = "ground_truth/pred.wav",
    pred_path: str = "stemgen_base/pred.wav",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Computes BeatTransformer's alignment scores for a specific generation folder.

    Args:
        generation_path (str): Path to a folder containing audio files.
        context_path (str): Relative path from generation_path to the context audio.
        gt_path (str): Relative path from generation_path to ground truth audio.
        pred_path (str): Relative path from generation_path to predicted audio.

    Returns:
        Tuple[Dict[float], float]: Ground truth (GT) score and predicted (PRED) scores.
    """
    context_path = os.path.join(generation_path, context_path)
    gt_path = os.path.join(generation_path, gt_path)
    pred_path = os.path.join(generation_path, pred_path)

    context_audio = preprocess_audio(context_path)
    gt_audio = preprocess_audio(gt_path)
    pred_audio = preprocess_audio(pred_path)

    with torch.no_grad():
        context_act, context_tempo = model(
            torch.from_numpy(context_audio).unsqueeze(0).float()
        )
        gt_act, gt_tempo = model(
            torch.from_numpy(gt_audio).unsqueeze(0).float()
        )
        pred_act, pred_tempo = model(
            torch.from_numpy(pred_audio).unsqueeze(0).float()
        )

    gt_score = {
        "bt_act_bce": torch.nn.functional.binary_cross_entropy_with_logits(
            torch.sigmoid(context_act),
            torch.sigmoid(gt_act),
            pos_weight=torch.LongTensor([1, 1]),
        ),
        "bt_tempo_bce": torch.nn.functional.binary_cross_entropy(
            torch.softmax(context_tempo, dim=-1),
            torch.softmax(gt_tempo, dim=-1),
        ),
    }
    pred_score = {
        "bt_act_bce": torch.nn.functional.binary_cross_entropy_with_logits(
            torch.sigmoid(context_act),
            torch.sigmoid(pred_act),
            pos_weight=torch.LongTensor([1, 1]),
        ),
        "bt_tempo_bce": torch.nn.functional.binary_cross_entropy(
            torch.softmax(context_tempo, dim=-1),
            torch.softmax(pred_tempo, dim=-1),
        ),
    }

    context_beat = beat_tracker(
        torch.sigmoid(context_act[0, :, 0]).detach().cpu().numpy()
    )
    gt_beat = beat_tracker(
        torch.sigmoid(gt_act[0, :, 0]).detach().cpu().numpy()
    )
    pred_beat = beat_tracker(
        torch.sigmoid(pred_act[0, :, 0]).detach().cpu().numpy()
    )

    try:
        # To avoid the case where the pred_beat is empty or only contains one beat
        gt_madmom_scores = BeatEvaluation(context_beat, gt_beat)
        pred_madmom_scores = BeatEvaluation(context_beat, pred_beat)
        keys = ["fmeasure", "cemgil", "cmlc", "cmlt", "amlc", "amlt"]

        # update scores with madmom evaluation
        gt_score.update(
            {f"madmom_{key}": getattr(gt_madmom_scores, key) for key in keys}
        )
        pred_score.update(
            {f"madmom_{key}": getattr(pred_madmom_scores, key) for key in keys}
        )
    except Exception as e:
        print(f"Error in madmom evaluation: {e}")
        gt_madmom_scores = {}
        pred_madmom_scores = {}

    return gt_score, pred_score


def compute_beat_alignment_given_path_beat_this(
    generation_path: str,
    context_path: str = "input_audio.wav",
    gt_path: str = "ground_truth/pred.wav",
    pred_path: str = "stemgen_base/pred.wav",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Computes BeatTransformer's alignment scores for a specific generation folder.

    Args:
        generation_path (str): Path to a folder containing audio files.
        context_path (str): Relative path from generation_path to the context audio.
        gt_path (str): Relative path from generation_path to ground truth audio.
        pred_path (str): Relative path from generation_path to predicted audio.

    Returns:
        Tuple[Dict[float], float]: Ground truth (GT) score and predicted (PRED) scores.
    """
    context_path = os.path.join(generation_path, context_path)
    gt_path = os.path.join(generation_path, gt_path)
    pred_path = os.path.join(generation_path, pred_path)

    gt_beat, gt_downbeats = file2beats(gt_path)
    pred_beat, pred_downbeats = file2beats(pred_path)
    context_beat, context_downbeats = file2beats(context_path)

    gt_score = {}
    pred_score = {}

    try:
        # To avoid the case where the pred_beat is empty or only contains one beat
        gt_madmom_scores = BeatEvaluation(context_beat, gt_beat)
        pred_madmom_scores = BeatEvaluation(context_beat, pred_beat)
        keys = ["fmeasure", "cemgil", "cmlc", "cmlt", "amlc", "amlt"]

        # update scores with madmom evaluation
        gt_score.update(
            {f"madmom_{key}": getattr(gt_madmom_scores, key) for key in keys}
        )
        pred_score.update(
            {f"madmom_{key}": getattr(pred_madmom_scores, key) for key in keys}
        )
    except Exception as e:
        print(f"Error in madmom evaluation: {e}")
        gt_madmom_scores = {}
        pred_madmom_scores = {}

    return gt_score, pred_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute BeatTransformer alignment scores."
    )
    parser.add_argument(
        "root_dir", type=str, help="Root dir containing examples."
    )
    parser.add_argument(
        "--context_path",
        type=str,
        default="input_audio.wav",
        help="Relative path to context audio.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="ground_truth/pred.wav",
        help="Relative path to ground truth audio.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="stemgen_large_8_rvq/pred.wav",
        help="Relative path to predicted audio.",
    )
    args = parser.parse_args()

    all_gt_scores, all_pred_scores = beat_alignment_score(
        args.root_dir,
        context_path=args.context_path,
        gt_path=args.gt_path,
        pred_path=args.pred_path,
    )

    print("Mean GT Scores:")
    for key, value in all_gt_scores.items():
        print(f"{key}: {np.mean(value):.4f}")
    print("\nMean Predicted Scores:")
    for key, value in all_pred_scores.items():
        print(f"{key}: {np.mean(value):.4f}")
