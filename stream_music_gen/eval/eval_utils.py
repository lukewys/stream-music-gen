import os
import math
import tqdm
import torch
import torchaudio

from typing import Optional


@torch.no_grad()
def resample(x: torch.Tensor, orig_fs: int, target_fs: int) -> torch.Tensor:
    """
    Resamples the input audio tensor to the target sampling rate.

    Args:
        x (torch.Tensor): Input audio tensor of shape
            (num_channels, num_samples) or (num_samples).
        orig_fs (int): Original sampling rate of the audio.
        target_fs (int): Target sampling rate to resample the audio to.

    Returns:
        torch.Tensor: Resampled audio tensor with updated sampling rate.

    Notes:
        Params from
        https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html
        Under "kaiser_best". Found experimentally to give clean spectrograms
        compared to default.
    """
    return torchaudio.functional.resample(
        waveform=x,
        orig_freq=orig_fs,
        new_freq=target_fs,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


def chunk_audio(
    x: torch.Tensor, frame_size: int, hop_length: int
) -> torch.Tensor:
    """
    Splits an audio tensor into overlapping frames.
    Zero-pads the last frame if necessary.

    Args:
        x (torch.Tensor): Input audio tensor of shape (num_samples,).
        frame_size (int): Number of samples per frame.
        hop_length (int): Number of samples between the start of consecutive frames.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, frame_size) containing the
                      overlapping audio frames. Each row corresponds to one frame.
    """
    # Pad to include last frame
    num_samples = x.shape[-1]

    if num_samples <= frame_size:
        num_frames = 1
    else:
        num_frames = math.ceil((num_samples - frame_size) / hop_length) + 1
    required_length = (num_frames - 1) * hop_length + frame_size

    padding_length = max(0, required_length - num_samples)
    if padding_length > 0:
        x = torch.nn.functional.pad(
            x, (0, padding_length), mode="constant", value=0
        )

    return x.unfold(dimension=-1, size=frame_size, step=hop_length)


def load_resample_save(
    paths: list[str],
    source_fs: int,
    target_fs: int,
    use_gpu: bool = True,
    batch_size: int = 512,
) -> None:
    """
    Resamples audio files to the target sample rates, saves
    the resampled audio with the sample rate as a suffix.

    Args:
        paths: paths to .wav files to load from
        source_fs: source sample rate
        target_fs: target sample rate
        use_gpu: flag for accelerating using GPU
        batch_size: number of audio samples to resample at a time
    """

    def process_buffer(path_buffer, load_buffer):

        assert len(path_buffer) == len(load_buffer)
        if len(load_buffer) == 0:
            return

        audios = torch.stack(load_buffer, dim=0)
        if use_gpu:
            audios = resample(audios.cuda(), source_fs, target_fs).cpu()
        else:
            audios = resample(audios, source_fs, target_fs)

        # Save Audio
        for j, audio in enumerate(audios):
            original_path = path_buffer[j]
            dir_path = os.path.dirname(original_path)
            filename = os.path.basename(original_path)
            name, ext = os.path.splitext(filename)

            new_filename = f"{name}_{target_fs}{ext}"
            new_path = os.path.join(dir_path, new_filename)
            torchaudio.save(new_path, audio, target_fs)

    path_buffer = []
    load_buffer = []

    for path in tqdm.tqdm(paths):
        x, fs = torchaudio.load(path)
        assert fs == source_fs, "sample rate of audio must match source_fs"

        if fs == target_fs:
            continue

        path_buffer.append(path)
        load_buffer.append(x)

        if len(load_buffer) >= batch_size:
            process_buffer(path_buffer, load_buffer)
            path_buffer.clear()
            load_buffer.clear()

    process_buffer(path_buffer, load_buffer)


def load_and_chunk(
    path: str,
    frame_size: int,
    hop_length: int,
    required_fs: Optional[int] = None,
) -> torch.Tensor:
    """
    Loads a .wav file and splits it into overlapping frames.

    Args:
        path (str): Path to the input .wav file.
        frame_size (int): Number of samples per frame.
        hop_length (int): Number of samples between the start of consecutive frames.
        required_fs (int): Optionally assert the sample rate to be a particular value.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, 1, frame_size) containing overlapping
        audio frames.

    Notes:
        Assumes the input audio is single-channel or mixes it down to mono if it has
        multiple channels.
    """
    x, fs = torchaudio.load(path)

    if required_fs is not None:
        assert fs == required_fs

    if x.dim() > 1:
        x = x.mean(dim=0)
    chunks = chunk_audio(x, frame_size, hop_length).unsqueeze(1)
    return chunks
