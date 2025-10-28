"""Utility functions for plotting images."""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd


def audio_to_spectrogram_image(wav, sr, vmin=-8, vmax=1, cmap="magma"):
    """
    Convert a 1D audio waveform into an RGB spectrogram image array.

    Parameters:
    -----------
    wav : np.ndarray
        1D numpy array of the audio waveform.
    sr : int
        Sampling rate of the audio.
    vmin : float, optional
        Minimum dB range for the spectrogram display.
    vmax : float, optional
        Maximum dB range for the spectrogram display.
    cmap : str, optional
        Colormap to use for the spectrogram.

    Returns:
    --------
    image : np.ndarray
        A 3D numpy array (H x W x 3) representing the RGB spectrogram image.
    """

    # Compute the STFT and take the log of the magnitude
    D = np.log(
        np.abs(librosa.stft(wav, n_fft=768)) + 1e-9
    )  # add small epsilon to avoid log(0)

    # Create a figure and plot the spectrogram
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    librosa.display.specshow(D, sr=sr, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax)
    ax.set_title("")
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Draw the figure and convert it to a numpy array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape(
        height, width, 3
    )

    # Close the figure to free memory
    plt.close(fig)

    return image


def plot_spec(wav, sr, title="", play=True, vmin=-8, vmax=1, save_path=None):
    """Plot mel spectrogram of a waveform in notebook."""
    D = np.log(np.abs(librosa.stft(wav, n_fft=512 + 256)))
    librosa.display.specshow(D, sr=sr, vmin=vmin, vmax=vmax, cmap="magma")
    plt.title(title)
    wav = np.clip(wav, -1, 1)
    if play:
        ipd.display(ipd.Audio(wav, rate=sr))
    if save_path:
        plt.savefig(save_path)
        plt.close()
