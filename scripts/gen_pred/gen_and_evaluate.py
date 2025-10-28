import argparse
import argbind
import os
import numpy as np
import glob
import json
import shutil
from lightning import seed_everything
import stream_music_gen.eval.eval_utils as eval_utils
import stream_music_gen.eval.cocola_eval as cocola_eval
import stream_music_gen.eval.fad_eval as fad_eval
from stream_music_gen.eval.beat_alignment_eval import beat_alignment_score
from stream_music_gen.constants import DAC_SAMPLE_RATE, EVAL_SAMPLE_RATE

import warnings

warnings.filterwarnings("ignore", module="pyloudnorm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation Main Script",
    )
    parser.add_argument(
        "--model_type",
        help="enc_dec_flatten, enc_dec_multiout, stemgen, stemgen_large_8_rvq, dec_online, random_anchor, prefix_decoder",
    )
    parser.add_argument("--model_path", help="path to checkpoint")
    parser.add_argument(
        "--skip_audio_generation",
        help="set to true if audio has been generated already",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--skip_resampling",
        help="set to true to skip resampling to 16 khz",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_beat_alignment",
        help="set to true to skip beat alignment score calculation",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--skip_cocola",
        help="set to true to skip cocola score calculation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_fad",
        help="set to true to skip fad calculation",
        action="store_true",
        default=False,
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_samples",
        help="number of samples to generate",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--sub_fad",
        help="Evaluate FAD on mixes (sub-FAD) instead of on generated stems",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data_base_dir",
        help="path to data base directory, if specified, will override the data_base_dir in the config file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gen_kwargs",
        help="kwargs for audio generation",
        type=str,
        default="{}",
    )
    parser.add_argument(
        "--save_dir_name",
        help="name of the save directory",
        type=str,
        default="model_predictions",
    )
    parser.add_argument(
        "--audio_save_dir",
        help="directory to save generated audio files",
        type=str,
        default="",
    )
    parser.add_argument(
        "--results_save_dir",
        help="directory to save evaluation results JSON files",
        type=str,
        default="online-stem-gen/logs/gen_eval_results",
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--remove_generation",
        help="remove all generated files and folders after evaluation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--audio_generation_only",
        help="only generate audio, skip evaluation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--split",
        help="dataset split to use for generation",
        type=str,
        default="valid",
        choices=["valid", "test", "train"],
    )
    args = parser.parse_args()

    gen_kwargs = json.loads(args.gen_kwargs)

    if args.seed is not None:
        seed_everything(args.seed)

    # Load model config, data directories.

    ckpt_dir = os.path.dirname(args.model_path)

    if args.model_type != "random_anchor":
        config = argbind.load_args(os.path.join(ckpt_dir, "args.yml"))
        data_base_dir = args.data_base_dir or config["data_base_dir"]
    else:
        config = None
        data_base_dir = args.data_base_dir

    # Read model type
    if args.model_type == "stemgen":
        from gen_pred_stemgen import main
    elif args.model_type == "stemgen_large_8_rvq":
        from gen_pred_stemgen_large_8_rvq import main
    elif args.model_type == "dec_online":
        from gen_pred_dec_online import main
    elif args.model_type == "random_anchor":
        from gen_random_anchor import main
    elif args.model_type == "prefix_decoder":
        from gen_pred_prefix_decoder import main
    elif args.model_type == "prefix_decoder_online":
        from gen_pred_prefix_dec_online import main

    if args.audio_save_dir == "":
        root_folder = os.path.join(ckpt_dir, args.save_dir_name)
    else:
        root_folder = os.path.join(args.audio_save_dir, args.save_dir_name)

    # Initialize results dictionary
    results = {}

    # Generate Audio
    if not args.skip_audio_generation:
        main(
            model_path=args.model_path,
            save_dir=root_folder,
            data_base_dir=data_base_dir,
            batch_size=args.batch_size,
            model_name="pred",
            num_samples=args.num_samples,
            gen_kwargs=gen_kwargs,
            split=args.split,
        )

    if args.audio_generation_only:
        print(f"Audio generation only, skipping evaluation")
        exit()

    if not args.skip_resampling:
        # Resample Audio, and save
        files_to_resample = glob.glob(
            os.path.join(root_folder, "**", "*.wav"), recursive=True
        )
        eval_utils.load_resample_save(
            files_to_resample,
            DAC_SAMPLE_RATE,
            EVAL_SAMPLE_RATE,
        )

    if not args.skip_beat_alignment:
        # Evaluate Audio - Beat Alignment (f_measure only)
        all_gt_scores, all_pred_scores = beat_alignment_score(
            root_folder,
            context_path="input_audio.wav",
            gt_path="ground_truth/pred.wav",
            pred_path="pred/pred.wav",
        )

        # Extract f_measure only
        f_measure_key = "madmom_fmeasure"
        if f_measure_key in all_gt_scores:
            gt_f_measure = all_gt_scores[f_measure_key]
            pred_f_measure = all_pred_scores[f_measure_key]

            results["beat_alignment"] = {
                "gt_f_measure": {
                    "mean": float(np.mean(gt_f_measure)),
                    "std": float(np.std(gt_f_measure)),
                },
                "pred_f_measure": {
                    "mean": float(np.mean(pred_f_measure)),
                    "std": float(np.std(pred_f_measure)),
                },
            }

            print("Beat Alignment F-Measure Scores:")
            print(
                f"GT F-Measure - Mean: {np.mean(gt_f_measure):.4f}, Std: {np.std(gt_f_measure):.4f}"
            )
            print(
                f"Pred F-Measure - Mean: {np.mean(pred_f_measure):.4f}, Std: {np.std(pred_f_measure):.4f}"
            )
        else:
            print(
                f"Warning: {f_measure_key} not found in beat alignment results"
            )
            results["beat_alignment"] = {"error": f"{f_measure_key} not found"}

    if not args.skip_cocola:
        # Evaluate Audio - COCOLA
        embedding_modes = ["both", "harmonic", "percussive"]

        gt_scores, pred_scores = cocola_eval.cocola_score(
            root_folder,
            context_path=f"input_audio_{EVAL_SAMPLE_RATE}.wav",
            gt_path=f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav",
            pred_path=f"pred/pred_{EVAL_SAMPLE_RATE}.wav",
            embedding_modes=embedding_modes,
        )

        results["cocola"] = {}
        for embedding_mode in embedding_modes:
            gt_scores_mode = gt_scores[embedding_mode]
            pred_scores_mode = pred_scores[embedding_mode]

            results["cocola"][embedding_mode] = {
                "gt_scores": {
                    "mean": float(np.mean(gt_scores_mode)),
                    "std": float(np.std(gt_scores_mode)),
                },
                "pred_scores": {
                    "mean": float(np.mean(pred_scores_mode)),
                    "std": float(np.std(pred_scores_mode)),
                },
            }

            print(f"CoCoLa Scores, Embedding Mode={embedding_mode}:")
            print(
                f"  GT - Mean: {np.mean(gt_scores_mode):.4f}, Std: {np.std(gt_scores_mode):.4f}"
            )
            print(
                f"  Pred - Mean: {np.mean(pred_scores_mode):.4f}, Std: {np.std(pred_scores_mode):.4f}"
            )

    if not args.skip_fad:
        # Evaluate audio - FAD
        if args.sub_fad:
            gt_path = f"ground_truth/mix_{EVAL_SAMPLE_RATE}.wav"
            gen_path = f"pred/mix_{EVAL_SAMPLE_RATE}.wav"
        else:
            gt_path = f"ground_truth/pred_{EVAL_SAMPLE_RATE}.wav"
            gen_path = f"pred/pred_{EVAL_SAMPLE_RATE}.wav"

        fad_score = fad_eval.calculate_fad(
            root_folder,
            gt_path=gt_path,
            gen_path=gen_path,
            metric="vggish",
            verbose=False,
        )

        results["fad"] = {"score": float(fad_score)}
        print(f"FAD Score: {fad_score}")

    # Save results to JSON
    os.makedirs(args.results_save_dir, exist_ok=True)
    results_file = os.path.join(
        args.results_save_dir, args.save_dir_name + ".json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Remove generation files if requested
    if args.remove_generation:
        print(f"\nRemoving generation files from: {root_folder}")
        # Remove all numbered directories and their contents
        for item in glob.glob(os.path.join(root_folder, "*")):
            if os.path.isdir(item) and os.path.basename(item).isdigit():
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
        # Remove any remaining wav files and metadata.json files
        for item in glob.glob(os.path.join(root_folder, "*.wav")):
            os.remove(item)
            print(f"Removed file: {item}")
        for item in glob.glob(os.path.join(root_folder, "*.json")):
            if (
                os.path.basename(item) != "evaluation_results.json"
            ):  # Keep the old one for now
                os.remove(item)
                print(f"Removed file: {item}")
        print("Generation files cleanup completed.")
