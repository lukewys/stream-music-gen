import os
import glob
import argparse
import soundfile as sf
from tqdm import tqdm
from stream_music_gen.utils.audio_utils import loudness_normalize_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Audio.")
    parser.add_argument(
        "--audio_pattern",
        type=str,
        help="pattern of audio files to normalize",
        default="models/reorganized_examples/**/*.wav",
    )
    args = parser.parse_args()

    paths = sorted(
        glob.glob(
            args.audio_pattern,
            recursive=True,
        )
    )

    for path in tqdm(paths):
        print(path)
        audio, sr = sf.read(path)
        normalized = loudness_normalize_audio(
            audio,
            32000,
        )
        out_path = path.replace(".wav", "_normalized.wav")
        sf.write(out_path, normalized, sr)
        os.remove(path)
