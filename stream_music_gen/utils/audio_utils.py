"""Audio utilities."""

import numpy as np
import pyloudnorm as pyln

DB_TARGET = -18.0  # target loudness for the audio
PRED_DB_OFFSET = 5.0  # offset to apply to the predicted stem in dB
PRED_DB_OFFSET_LOUD = 12.0  # offset to apply to the predicted stem in dB, if want to make pred louder
PEAK_TARGET = 0.99  # target peak level


def loudness_normalize_audio(
    audio: np.ndarray,
    sample_rate: int,
    db_target: float = DB_TARGET,
    peak_target: float = PEAK_TARGET,
):
    """Normalize the audio to the target loudness.

    Args:
        audio: numpy array of the audio.
        sample_rate: sample rate of the audio.
        db_target: target loudness in dB.
        peak_target: target peak level.
    Returns:
        normalized_audio: numpy array of the normalized audio.
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    loudness_normalized_audio = pyln.normalize.loudness(
        audio, loudness, db_target
    )
    peak = np.max(np.abs(loudness_normalized_audio))
    if peak > peak_target:
        loudness_normalized_audio *= peak_target / peak

    return loudness_normalized_audio


def mix_with_generated_stem(
    audio_input_mix: np.ndarray,
    pred_stem: np.ndarray,
    num_stems: int,
    sample_rate: int,
    db_target: float = DB_TARGET,
    pred_db_offset: float = PRED_DB_OFFSET,
    peak_target: float = PEAK_TARGET,
):
    """Return a mix that (i) embeds the generated stem at the correct
    perceptual level, (ii) averages amplitudes for N+1 stems, and
    (iii) (optionally) preserves integrated loudness.

    Args:
        audio_input_mix: numpy array of the input mix.
        pred_stem: numpy array of the predicted stem.
        num_stems: number of stems in the mix.
        sample_rate: sample rate of the audio.
        db_target: target loudness in dB.
        pred_db_offset: offset to apply to the predicted stem in dB.
        peak_target: target peak level.

    Returns:
        mixed: numpy array of the mixed audio.
    """
    meter = pyln.Meter(sample_rate)

    # 1. Match the stem’s loudness to what one source in the mix would have
    L_mix = meter.integrated_loudness(audio_input_mix)
    L_pred = meter.integrated_loudness(pred_stem)

    # Each individual stem is ~ L_mix - 10*log10(N)
    target_stem_lufs = L_mix - 10 * np.log10(num_stems)
    target_stem_lufs += pred_db_offset
    gain_lin = 10 ** ((target_stem_lufs - L_pred) / 20)
    stem_adj = pred_stem * gain_lin

    # 2. Add the new stem (energy basis)
    mixed = audio_input_mix + stem_adj

    # 3. Loudness normalise the whole mix to the final target
    L_current = meter.integrated_loudness(mixed)
    mixed = pyln.normalize.loudness(mixed, L_current, db_target)

    # 4. True-peak guard (using sample-peak here; oversample in production)
    peak = np.max(np.abs(mixed))
    if peak > peak_target:
        mixed *= peak_target / peak

    return mixed
