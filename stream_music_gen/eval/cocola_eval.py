import os
import argparse
import torch
import numpy as np
from stream_music_gen.eval.eval_utils import load_and_chunk
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
from pqdm.processes import pqdm

# Import cocola modules
import contrastive_model
from contrastive_model import constants
from contrastive_model.contrastive_model import CoCola
from feature_extraction.feature_extraction import CoColaFeatureExtractor

# Constants
CHUNK_SIZE = 16000 * 5
HOP_LENGTH = 8000 * 5
FS = 16000  # Target FS for Cocola - this should be kept at 16 khz.

# Download the COCOLA checkpoint here:
# https://drive.google.com/file/d/1S-_OvnDwNFLNZD5BmI1Ouck_prutRVWZ/view
# Then, place it in the cocola_models/ directory.
MODEL_PATH = "cocola_models/checkpoint-epoch=87-val_loss=0.00.ckpt"


torch.manual_seed(0)


def extract_hpss_features(
    x: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    win_length: int = 400,
    hop_length: int = 160,
    f_min: float = 60.0,
    f_max: float = 7800.0,
    n_mels: int = 64,
) -> np.ndarray:
    """
    Function-based HPSS feature extraction to avoid nn.Module deadlock issues with pqdm.

    Args:
        x (np.ndarray): The audio tensor(s) of shape (B, 1, S) or (1, S).
        sample_rate (int): Sample rate of the audio.
        n_fft (int): FFT window size.
        win_length (int): Window length for STFT.
        hop_length (int): Hop length for STFT.
        f_min (float): Minimum frequency for mel spectrogram.
        f_max (float): Maximum frequency for mel spectrogram.
        n_mels (int): Number of mel bands.

    Returns:
        np.ndarray: The HPSS features tensor(s) of shape (B, 2, H, W) or (2, H, W).
    """
    import librosa

    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    batch_size = x.shape[0]
    features = []
    for i in range(batch_size):
        audio = x[i].squeeze(0)
        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        harmonic_stft, percussive_stft = librosa.decompose.hpss(stft)
        mel_harmonic = librosa.feature.melspectrogram(
            S=np.abs(harmonic_stft) ** 2,
            sr=sample_rate,
            fmin=f_min,
            fmax=f_max,
            n_mels=n_mels,
        )
        mel_percussive = librosa.feature.melspectrogram(
            S=np.abs(percussive_stft) ** 2,
            sr=sample_rate,
            fmin=f_min,
            fmax=f_max,
            n_mels=n_mels,
        )
        mel_db_harmonic = librosa.power_to_db(mel_harmonic, ref=np.max)
        mel_db_percussive = librosa.power_to_db(mel_percussive, ref=np.max)
        hp_mel_db = np.stack((mel_db_harmonic, mel_db_percussive), axis=0)
        features.append(hp_mel_db)

    features = np.stack(features, axis=0)
    if batch_size == 1:
        features = features.squeeze(0)

    return features


def _cocola_score_old(
    root_folder: str,
    context_path: str = "input_audio_16000.wav",
    gt_path: str = "ground_truth/pred_16000.wav",
    pred_path: str = "pred/pred_16000.wav",
    embedding_mode: str = "both",
    device: Optional[str] = "cuda:0",
) -> Tuple[list, list]:
    """
    Computes CoCoLa scores for all generation folders in the root directory.
    This function is non-parallelized, old version.

    Args:
        root_folder (str): Path to the root directory containing folders of
            generation examples.
        context_path (str): Relative path from single example folder
            to the context audio
        gt_path (str): Relative path from single example folder to
            ground truth audio.
        pred_path (str): Relative path from single example folder
            to predicted audio
        embedding_mode (str): COCOLA Embedding mode - harmonic, percussive, or both.
        device (str): what torch device to place the model on.

    Returns:
        Tuple[List[float], List[float]]: Lists of ground truth (GT) scores and predicted (PRED) scores.
    """
    device = torch.device(device if device is not None else "cpu")

    # Load Model
    model = CoCola.load_from_checkpoint(MODEL_PATH, map_location=device)
    feature_extractor = CoColaFeatureExtractor()
    model.eval()
    feature_extractor.to(device)

    if embedding_mode == "both":
        model.set_embedding_mode(constants.EmbeddingMode.BOTH)
    elif embedding_mode == "harmonic":
        model.set_embedding_mode(constants.EmbeddingMode.HARMONIC)
    elif embedding_mode == "percussive":
        model.set_embedding_mode(constants.EmbeddingMode.PERCUSSIVE)
    else:
        raise ValueError("Invalid embedding_mode")

    # Get each subfolder in root_folder
    generation_paths = sorted(
        [
            os.path.join(root_folder, d)
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
    )
    gt_scores = []
    pred_scores = []

    # Compute cocola within each subfolder
    for generation_path in tqdm(
        generation_paths, desc="Processing CoCoLa Scores"
    ):
        gt_score, pred_score = compute_cocola_given_path(
            generation_path,
            context_path,
            gt_path,
            pred_path,
            model,
            feature_extractor,
        )
        gt_scores.append(gt_score)
        pred_scores.append(pred_score)

    return gt_scores, pred_scores


def _extract_chunks_for_path(
    generation_path: str,
    context_path: str,
    gt_path: str,
    pred_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to load audio, chunk, and extract CoCoLa features."""

    context_full = os.path.join(generation_path, context_path)
    gt_full = os.path.join(generation_path, gt_path)
    pred_full = os.path.join(generation_path, pred_path)

    context_chunks = load_and_chunk(context_full, CHUNK_SIZE, HOP_LENGTH, FS)
    gt_chunks = load_and_chunk(gt_full, CHUNK_SIZE, HOP_LENGTH, FS)
    pred_chunks = load_and_chunk(pred_full, CHUNK_SIZE, HOP_LENGTH, FS)

    return (
        context_chunks.cpu().numpy(),
        gt_chunks.cpu().numpy(),
        pred_chunks.cpu().numpy(),
    )


def _extract_features_for_chunks(
    context_chunks: np.ndarray,
    gt_chunks: np.ndarray,
    pred_chunks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper to load audio, chunk, and extract CoCoLa features."""
    with torch.no_grad():
        feat_context = extract_hpss_features(context_chunks)
        feat_gt = extract_hpss_features(gt_chunks)
        feat_pred = extract_hpss_features(pred_chunks)

    return feat_context, feat_gt, feat_pred


def cocola_score(
    root_folder: str,
    context_path: str = "input_audio_16000.wav",
    gt_path: str = "ground_truth/pred_16000.wav",
    pred_path: str = "pred/pred_16000.wav",
    embedding_modes: List[str] = ["both"],
    n_jobs: Optional[int] = None,
    device: Optional[str] = "cuda:0",
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Compute CoCoLa scores using parallel feature extraction."""

    torch.cuda.empty_cache()

    device = torch.device(device if device is not None else "cpu")
    if n_jobs is None:
        n_jobs = os.cpu_count() // 2

    model = CoCola.load_from_checkpoint(MODEL_PATH, map_location=device)
    model.eval()

    generation_paths = sorted(
        [
            os.path.join(root_folder, d)
            for d in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, d))
        ]
    )

    chunks_all = []

    for generation_path in tqdm(
        generation_paths, desc="Loading and Chunking Audio"
    ):
        context_chunks, gt_chunks, pred_chunks = _extract_chunks_for_path(
            generation_path, context_path, gt_path, pred_path
        )
        chunks_all.append((context_chunks, gt_chunks, pred_chunks))

    features_all = pqdm(
        chunks_all,
        _extract_features_for_chunks,
        n_jobs=n_jobs,
        argument_type="args",
        desc="Extracting CoCoLa Features",
    )

    gt_scores: Dict[str, List[float]] = {}
    pred_scores: Dict[str, List[float]] = {}

    for embedding_mode in embedding_modes:
        gt_scores[embedding_mode] = []
        pred_scores[embedding_mode] = []

        if embedding_mode == "both":
            model.set_embedding_mode(constants.EmbeddingMode.BOTH)
        elif embedding_mode == "harmonic":
            model.set_embedding_mode(constants.EmbeddingMode.HARMONIC)
        elif embedding_mode == "percussive":
            model.set_embedding_mode(constants.EmbeddingMode.PERCUSSIVE)
        else:
            raise ValueError("Invalid embedding_mode")

        for feat_context, feat_gt, feat_pred in tqdm(
            features_all, desc="Computing CoCoLa Scores"
        ):
            with torch.no_grad():
                feat_context = torch.from_numpy(feat_context).to(model.device)
                feat_gt = torch.from_numpy(feat_gt).to(model.device)
                feat_pred = torch.from_numpy(feat_pred).to(model.device)
                gt_val = model.score(feat_context, feat_gt)
                pred_val = model.score(feat_context, feat_pred)

            gt_scores[embedding_mode].append(torch.mean(gt_val).item())
            pred_scores[embedding_mode].append(torch.mean(pred_val).item())

    torch.cuda.empty_cache()

    return gt_scores, pred_scores


def compute_cocola_given_path(
    generation_path: str,
    context_path: str = "input_audio_16000.wav",
    gt_path: str = "ground_truth/pred_16000.wav",
    pred_path: str = "pred/pred_16000.wav",
    model: Optional[CoCola] = None,
    feature_extractor: Optional[CoColaFeatureExtractor] = None,
) -> Tuple[float, float]:
    """
    Computes CoCoLa scores for a specific generation folder.

    Args:
        generation_path (str): Path to a folder containing audio files
        context_path (str): Relative path from generation_path to the context audio
        gt_path (str): Relative path from generation_path to ground truth audio.
        pred_path (str): Relative path from generation_path to predicted audio
        model (Optional[CoCola]): Cocola model
        feature_extractor (Optional[CoColaFeatureExtractor]): Cocola feature extractor

    Returns:
        Tuple[float, float]: Ground truth (GT) score and predicted (PRED) score.
    """

    context_path = os.path.join(generation_path, context_path)
    gt_path = os.path.join(generation_path, gt_path)
    pred_path = os.path.join(generation_path, pred_path)

    context_chunks = load_and_chunk(context_path, CHUNK_SIZE, HOP_LENGTH, FS)
    gt_chunks = load_and_chunk(gt_path, CHUNK_SIZE, HOP_LENGTH, FS)
    pred_chunks = load_and_chunk(pred_path, CHUNK_SIZE, HOP_LENGTH, FS)

    gt_score = torch.mean(
        compute_cocola_distance(
            context_chunks, gt_chunks, model, feature_extractor
        )
    ).item()
    pred_score = torch.mean(
        compute_cocola_distance(
            context_chunks, pred_chunks, model, feature_extractor
        )
    ).item()

    return gt_score, pred_score


def compute_cocola_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    model: CoCola,
    feature_extractor: CoColaFeatureExtractor,
) -> torch.Tensor:
    """
    Computes CoCoLa distance between corresponding frames of two audio tensors.

    Args:
        x (torch.Tensor): Tensor of shape (B, 1, CHUNK_SIZE), where B is the batch size.
        y (torch.Tensor): Tensor of shape (B, 1, CHUNK_SIZE), where B is the batch size.
        model (CoCola): Cocola model
        feature_extractor (CoColaFeatureExtractor): Cocola feature extractor

    Returns:
        torch.Tensor: Tensor of CoCoLa scores for each pair of frames.
    """
    with torch.no_grad():
        features_x = feature_extractor(x)
        features_y = feature_extractor(y)
        scores = model.score(
            features_x.to(model.device), features_y.to(model.device)
        )
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CoCoLa scores.")
    parser.add_argument(
        "root_dir", type=str, help="Root dir containing examples."
    )
    parser.add_argument(
        "--mode", type=str, default="both", help="CoCoLa Embedding Mode"
    )

    parser.add_argument(
        "--context_path",
        type=str,
        default="input_audio_16000.wav",
        help="Relative path to context audio.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="ground_truth/pred_16000.wav",
        help="Relative path to ground truth audio.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="pred/pred_16000.wav",
        help="Relative path to predicted audio.",
    )
    args = parser.parse_args()

    gt_scores, pred_scores = cocola_score(
        args.root_dir,
        context_path=args.context_path,
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        embedding_mode=args.mode,
    )

    # np.save(os.path.join(args.root_dir, "cocola_gt_scores.npy"), gt_scores)
    # np.save(os.path.join(args.root_dir, "cocola_pred_scores.npy"), pred_scores)

    print(f"Mean CoCoLa GT Score: {np.mean(gt_scores)}")
    print(f"Mean CoCoLa Pred Score: {np.mean(pred_scores)}")
