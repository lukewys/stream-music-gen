"""Transforms for dataloader."""

from pathlib import Path
from typing import List, Optional
import random
import copy
import torch
import torchaudio
import torchaudio.transforms as T


class BaseTransform:
    """Base class for transforms."""

    def __call__(self, item):
        return item


class ComposeTransform:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, item):
        for transform in self.transforms:
            item = transform(item)
        return item


class SelectStems(BaseTransform):
    """
    Select stems from a track.

    Args:
        allow_instrument_ids (List[int]): If provided, only these instrument IDs are allowed
            for the target stem.
        ignore_instrument_ids (List[int]): If provided, these instrument IDs are NOT allowed
            for the target stem.
        num_rvq_layers (int): Number of RVQ layers (used to slice tokens).
        load_audio (bool): Whether to load audio waveforms.
        audio_base_dir (str): Base directory for audio files.
        duration (int): Duration of the audio crop in seconds.
        target_sample_rate (int): Sample rate to resample audio to if load_audio is True.
        token_crop_length (int): Number of frames (tokens) to crop from each stem.
        filter_stem_by_rms (bool): If True, attempts to align crops so that non-silent frames
            of input and target overlap.
        rms_threshold (float): Loudness threshold in dB for counting a frame as non-silent.
        num_retries (int): Number of times to retry random selection before giving up.
        min_overlap_portion (float): Minimum fraction of frames that must be non-silent in the
            overlap region.
        copy_target_to_input (bool): If True, uses the target stem as the input stem.
    """

    def __init__(
        self,
        allow_instrument_ids: List[int] = None,
        ignore_instrument_ids: List[int] = None,
        num_rvq_layers: int = 8,
        load_audio: bool = False,
        audio_base_dir: str = None,
        duration: int = 10,
        target_sample_rate: int = 32000,
        token_crop_length: int = 500,  # number of token frames
        max_num_input_stems: Optional[int] = None,
        filter_stem_by_rms: bool = True,
        rms_threshold: float = -60.0,
        num_retries: int = 10,
        min_overlap_portion: float = 0.5,
        rms_token_hz: int = 50,
        copy_target_to_input: bool = False,
        load_audio_tokens: bool = True,
    ):
        if (
            allow_instrument_ids is not None
            and ignore_instrument_ids is not None
        ):
            raise ValueError(
                "Only one of allow_instrument_ids or ignore_instrument_ids can be provided."
            )

        self.allow_instrument_ids = allow_instrument_ids
        self.ignore_instrument_ids = ignore_instrument_ids
        self.num_rvq_layers = num_rvq_layers
        self.load_audio = load_audio
        self.target_sample_rate = target_sample_rate
        self.token_crop_length = token_crop_length
        self.max_num_input_stems = max_num_input_stems
        self.duration = duration
        self.filter_stem_by_rms = filter_stem_by_rms
        self.rms_threshold = rms_threshold
        self.num_retries = num_retries
        self.min_overlap_portion = min_overlap_portion
        self.rms_token_hz = rms_token_hz
        if load_audio and audio_base_dir is None:
            raise ValueError(
                "audio_base_dir must be provided if load_audio is True."
            )
        self.audio_base_dir = Path(audio_base_dir)
        self.copy_target_to_input = copy_target_to_input
        self.load_audio_tokens = load_audio_tokens

    def filter_instrument_ids(self, instrument_ids):
        """
        Filter a list of instrument IDs according to allow_instrument_ids or ignore_instrument_ids.

        Args:
            instrument_ids (List[int]): All the instrument IDs from the stems.

        Returns:
            stem_indices (List[int]): Indices of stems that satisfy the filter.
        """
        if self.allow_instrument_ids is not None:
            # Keep only allowed
            stem_indices = [
                i
                for i, inst_id in enumerate(instrument_ids)
                if inst_id in self.allow_instrument_ids
            ]
        elif self.ignore_instrument_ids is not None:
            # Keep everything except the ignored
            stem_indices = [
                i
                for i, inst_id in enumerate(instrument_ids)
                if inst_id not in self.ignore_instrument_ids
            ]
        else:
            # No filtering
            stem_indices = list(range(len(instrument_ids)))
        return stem_indices

    def single_try_select(self, item, debug=False):
        """
        Attempt a single random selection of target stem (filtered by instrument)
        and a random subset of input stems (unfiltered). Then decide a crop start
        index that satisfies RMS overlap constraints.

        Returns:
            (input_idxs, target_idx, start), is_pass

            - input_idxs: List[int], the indices of chosen input stems
            - target_idx: int, the index of chosen target stem
            - start: int, the crop start frame
            - is_pass: bool, whether the selection meets the overlap criteria
        """
        instrument_ids = item["instrument_class_id"]
        n_stems = len(instrument_ids)

        # ------------------------------------------------------
        # 1) Choose a single target from allowed stems
        # ------------------------------------------------------
        candidate_target_indices = self.filter_instrument_ids(instrument_ids)
        if len(candidate_target_indices) < 1:
            # If no stems pass the filter, fall back to picking any
            candidate_target_indices = list(range(n_stems))
        target_idx = random.choice(candidate_target_indices)

        # ------------------------------------------------------
        # 2) Choose a random subset (>=1) of input stems from the rest, no filter
        # ------------------------------------------------------
        remaining_indices = list(range(n_stems))
        remaining_indices.remove(target_idx)

        if len(remaining_indices) == 0:
            # Edge case: only 1 stem in the track, fallback
            if debug:
                print(
                    "Only one stem found, cannot pick input stems differently."
                )
            return ([target_idx], target_idx, 0), True

        # At least 1 input, up to all
        if self.max_num_input_stems is not None:
            n_input = min(len(remaining_indices), self.max_num_input_stems)
        else:
            n_input = len(remaining_indices)
        input_idxs = random.sample(remaining_indices, n_input)

        # ------------------------------------------------------
        # 3) Check lengths & build OR mask for inputs
        # ------------------------------------------------------
        stem_rms_list = item["stem_rms"]  # List[Tensor], each shape [total_T]
        target_rms = stem_rms_list[target_idx]
        target_length = target_rms.size(0)

        # Build input OR mask
        input_stems_rms_all = [stem_rms_list[i] for i in input_idxs]

        min_length = min(
            [rms.size(0) for rms in input_stems_rms_all] + [target_length]
        )

        input_mask = torch.zeros(min_length, dtype=torch.bool)

        for idx in input_idxs:
            current_mask = stem_rms_list[idx] > self.rms_threshold
            input_mask = input_mask | current_mask[:min_length]

        # Check target length vs crop length
        if min_length <= self.token_crop_length:
            # no choice but to start at 0
            return (input_idxs, target_idx, 0), True

        if not self.filter_stem_by_rms:
            # If not filtering by RMS, just pick any random start
            start = random.randint(0, min_length - self.token_crop_length)
            return (input_idxs, target_idx, start), True

        # We do the AND with the target
        target_mask = target_rms > self.rms_threshold
        target_mask = target_mask[:min_length]
        overall_mask = input_mask & target_mask

        # Find valid indices for starting the crop
        valid_indices = torch.nonzero(overall_mask, as_tuple=True)[0]
        valid_indices = valid_indices[
            valid_indices < min_length - self.token_crop_length
        ]

        if valid_indices.size(0) == 0:
            # No valid start => fallback to random
            if debug:
                print(
                    "No valid start indices after RMS filtering, fallback to random."
                )
            start = random.randint(0, min_length - self.token_crop_length)
            return (input_idxs, target_idx, start), False

        # Randomly pick a start from valid_indices
        start = random.choice(valid_indices).item()
        end = start + self.token_crop_length

        # Check portion of non-silence frames
        non_silence_portion = overall_mask[start:end].float().mean()

        if non_silence_portion < self.min_overlap_portion:
            if debug:
                print(
                    f"Portion of non-silent frames is too low: {non_silence_portion:.3f}"
                )
            return (input_idxs, target_idx, start), False

        return (input_idxs, target_idx, start), True

    def select(self, item):
        """
        Try multiple times to select a valid (input_idxs, target_idx, start).
        Returns the first success or the last attempt if all fail.
        """
        final_output = ([0], 0, 0)  # fallback
        for _ in range(self.num_retries):
            (input_idxs, target_idx, start), is_pass = self.single_try_select(
                item
            )
            final_output = (input_idxs, target_idx, start)
            if is_pass:
                return final_output
        # If we exhaust retries without a good pick, return the last attempt
        # with a warning
        input_idxs, target_idx, _ = final_output
        item_dir = item["audio_token_path"][0]
        print(
            f"Warning: Could not find good overlap for item "
            f"{item_dir}."
            f"Using final fallback "
            f"(input_idxs={input_idxs}, target_idx={target_idx})."
        )
        return final_output

    def load_tokens(self, token_path, start=None, end=None):
        """
        Loads tokens from disk and slices them.

        Args:
            token_path (str): Path to a Torch tensor on disk.
            start (int): Start frame index for slicing.
            end (int): End frame index for slicing. If None, uses `start + self.token_crop_length`.

        Returns:
            token (Tensor): shape [num_rvq_layers, T_slice]
        """
        # If your torch.save had a "weights_only" option, adapt accordingly;
        # otherwise, just do a vanilla load:
        token = torch.load(token_path)

        # Now slice out only the top `self.num_rvq_layers`
        token = token[: self.num_rvq_layers]  # shape [num_rvq_layers, T_total]

        if start is not None:
            if end is None:
                end = start + self.token_crop_length
            token = token[:, start:end]
        return token  # shape [num_rvq_layers, T_slice]

    def load_audio_stem(self, audio_path, start):
        """
        Loads a single stem from `audio_path`, resamples to self.target_sample_rate,
        and optionally merges all channels to mono.

        Args:
            audio_path (str): path to audio file
            start (float): start second

        Returns:
            audio (Tensor): shape
                - [1, T_samples] if load_mono_audio = True
                - [channels, T_samples] otherwise
                at `self.target_sample_rate`.
        """
        audio, sr = torchaudio.load(audio_path)  # shape [channels, T_orig]

        # Crop
        start_sample = int(start * sr)
        length_sample = int(self.duration * sr)
        audio = audio[:, start_sample : start_sample + length_sample]
        return audio, sr

    def post_process_audio(self, audio, sr):
        # Resample if needed
        if sr != self.target_sample_rate:
            resample = T.Resample(
                orig_freq=sr, new_freq=self.target_sample_rate
            )
            audio = resample(audio)  # shape [channels, T_resampled]

        # merge channels to mono by taking the mean across channels
        if audio.size(0) > 1:
            audio = audio.mean(dim=0)  # shape [T_orig]
        else:
            audio = audio.squeeze(0)

        return audio

    def load_stems(self, item, input_idxs, target_idx, start):
        """
        Build the output dictionary containing tokens (and possibly audio).

        Args:
            item (dict): a dictionary with metadata, e.g.:
                  {
                    "base_dir": Path,
                    "instrument_class_id": [...],
                    "audio_token_path": [...],
                    "audio_path": [...],
                    "audio_sample_rate": 48000,
                    ...
                  }
            input_idxs (List[int]): indices of chosen input stems
            target_idx (int): index of chosen target stem
            start (int): crop start (token index)

        Returns:
            output_item (dict) with the following keys:
              - "base_dir": str
              - "input_inst_id": List[int]
              - "target_inst_id": int
              - "input_token_path": List[str]
              - "target_token_path": str
              - "input_token": Tensor [N_inputs, num_rvq_layers, T_crop]
              - "target_token": Tensor [num_rvq_layers, T_crop]
              - (optionally) "input_audio": Tensor [N_inputs, channels, T_samples] or
                [N_inputs, 1, T_samples]
              - (optionally) "target_audio": Tensor [channels, T_samples] or
                [1, T_samples]
        """
        instrument_ids = item["instrument_class_id"]

        output_item = {
            "base_dir": str(item["base_dir"]),
        }

        # -- Instrument IDs
        output_item["input_inst_id"] = [instrument_ids[i] for i in input_idxs]
        output_item["target_inst_id"] = instrument_ids[target_idx]

        # -- Token paths
        output_item["input_token_path"] = [
            item["audio_token_path"][i] for i in input_idxs
        ]
        output_item["target_token_path"] = item["audio_token_path"][target_idx]

        if self.load_audio_tokens:
            # -----------------------------------------------------
            # Load tokens for all input stems => shape [N_inputs, num_rvq_layers, T_crop]
            # -----------------------------------------------------
            input_tokens_list = []
            for idx in input_idxs:
                token_i = self.load_tokens(
                    item["audio_token_path"][idx], start=start
                )
                # token_i shape: [num_rvq_layers, T_crop]
                input_tokens_list.append(token_i)  # [num_rvq_layers, T_crop]

            # Concatenate along the 0th dimension => [N_inputs, num_rvq_layers, T_crop]
            output_item["input_token_stems"] = input_tokens_list

            # -----------------------------------------------------
            # Target token => shape [num_rvq_layers, T_crop]
            # -----------------------------------------------------
            output_item["target_token"] = self.load_tokens(
                item["audio_token_path"][target_idx], start=start
            )

        # -----------------------------------------------------
        # Optionally load audio
        # -----------------------------------------------------
        if self.load_audio:
            # input audio => shape [N_inputs, T_samples]
            input_audio_list = []
            for idx in input_idxs:
                audio_i_path = self.audio_base_dir / item["audio_path"][idx]
                audio_i, sr = self.load_audio_stem(
                    audio_i_path, start / self.rms_token_hz
                )
                input_audio_list.append(audio_i)

            # target audio => shape [T_samples]
            target_audio_path = (
                self.audio_base_dir / item["audio_path"][target_idx]
            )
            target_audio, sr = self.load_audio_stem(
                target_audio_path, start / self.rms_token_hz
            )

            # ensure all input audio have same length
            min_length = min(
                [audio.size(0) for audio in input_audio_list]
                + [target_audio.size(0)]
            )

            for i in range(len(input_audio_list)):
                input_audio_list[i] = input_audio_list[i][:min_length]
            target_audio = target_audio[:min_length]

            input_audio = torch.stack(input_audio_list, dim=0).mean(
                0
            )  # shape [N_inputs, C, T_samples] -> [C, T_samples]

            input_audio = self.post_process_audio(input_audio, sr)
            target_audio = self.post_process_audio(target_audio, sr)

            output_item["input_audio"] = input_audio
            # Final shape [T_samples]

            output_item["target_audio"] = target_audio

            output_item["input_audio_path"] = [
                str(self.audio_base_dir / item["audio_path"][idx])
                for idx in input_idxs
            ]
            output_item["target_audio_path"] = str(
                self.audio_base_dir / item["audio_path"][target_idx]
            )

        return output_item

    def __call__(self, item):
        """
        Main entry point. Called by the dataset or dataloader:

        1) Select input & target stems + start
        2) Load tokens (and possibly audio)
        3) Return dictionary with everything

        Returns:
            output_item (dict)
        """
        input_idxs, target_idx, start = self.select(item)

        # If we are copying target to input, only load target
        if self.copy_target_to_input:
            input_idxs = [target_idx]

        return self.load_stems(item, input_idxs, target_idx, start)


class LoadRMS(BaseTransform):
    def __init__(self, rms_base_dir: str):
        self.rms_base_dir = rms_base_dir

    def __call__(self, item):
        stem_rms = []
        token_base_dir = item["base_dir"]
        for audio_token_path in item["audio_token_path"]:
            rms_path = self.rms_base_dir / Path(audio_token_path).relative_to(
                token_base_dir
            )
            rms = torch.load(rms_path, weights_only=True)
            if rms.ndim == 2:
                rms = rms.mean(dim=0)
            stem_rms.append(rms)
        item["stem_rms"] = stem_rms
        return item


class PadTokens(BaseTransform):
    """
    Pad audio tokens to a fixed length
    Args:
        multilayer (bool): flag to indicate when we are loading non-flattened
        RVQ tokens for the target_tokens. Used for delay-patterning.
    """

    def __init__(
        self,
        input_length: int,
        pad_token_id: int = 0,
        key: str = "input_token",
        multilayer=False,
    ):
        self.input_length = input_length
        self.pad_token_id = pad_token_id
        self.key = key
        self.multilayer = multilayer

    def __call__(self, item):
        token = item[self.key]
        if not self.multilayer:
            token_mask = torch.ones(token.size(0), dtype=torch.bool)
            if token.size(0) >= self.input_length:
                token = token[: self.input_length]
                token_mask = token_mask[: self.input_length]
            else:
                pad_length = self.input_length - token.size(0)
                token = torch.cat(
                    [
                        token,
                        torch.full(
                            (pad_length,),
                            self.pad_token_id,
                            dtype=token.dtype,
                        ),
                    ],
                    dim=0,
                )
                token_mask = torch.cat(
                    [token_mask, torch.zeros(pad_length, dtype=torch.bool)],
                    dim=0,
                )
        else:
            num_layers, T = token.shape
            token_mask = torch.ones((num_layers, T), dtype=torch.bool)
            if T >= self.input_length:
                token = token[:, : self.input_length]
                token_mask = token_mask[:, : self.input_length]
            else:
                pad_length = self.input_length - T
                pad_tensor = torch.full(
                    (num_layers, pad_length),
                    self.pad_token_id,
                    dtype=token.dtype,
                )
                token = torch.cat([token, pad_tensor], dim=-1)

                pad_mask = torch.zeros(
                    (num_layers, pad_length), dtype=torch.bool
                )

                token_mask = torch.cat([token_mask, pad_mask], dim=-1)

        item[self.key] = token
        item[f"{self.key}_mask"] = token_mask
        return item


class PadEmbs(BaseTransform):
    """Pad embeddings to a fixed length."""

    def __init__(
        self,
        input_length: int,
        pad_value: float = 0.0,
        key: str = "input_emb",
    ):
        self.input_length = input_length
        self.pad_value = pad_value
        self.key = key

    def __call__(self, item):
        emb = item[self.key]  # [T, d]
        emb_mask = torch.ones(emb.size(0), dtype=torch.bool, device=emb.device)

        if emb.size(0) >= self.input_length:
            emb = emb[: self.input_length]
            emb_mask = emb_mask[: self.input_length]
        else:
            pad_length = self.input_length - emb.size(0)
            pad_tensor = torch.full(
                (pad_length, emb.size(1)),
                self.pad_value,
                dtype=emb.dtype,
                device=emb.device,
            )
            emb = torch.cat([emb, pad_tensor], dim=0)
            pad_mask = torch.zeros(
                pad_length, dtype=torch.bool, device=emb.device
            )
            emb_mask = torch.cat([emb_mask, pad_mask], dim=0)

        item[self.key] = emb
        item[f"{self.key}_mask"] = emb_mask
        return item


class CodecTokensToLMTokens(BaseTransform):
    """Convert audio tokens to LM tokens."""

    def __init__(
        self,
        num_special_tokens: int,
        num_instrument_tokens: int,
        num_codebook_per_layer: int,
        shared: bool = False,
    ):
        """
        Args:
            shared (bool): flag to indicate if different RVQ levels should have
            the same range of ids. Should only be used with multilayer.
        """
        self.num_special_tokens = num_special_tokens
        self.num_instrument_tokens = num_instrument_tokens
        self.num_codebook_per_layer = num_codebook_per_layer
        self.shared = shared

    def codec_tokens_to_tokens(
        self, codec_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Convert codec tokens to tokens used by language model."""
        # codec_tokens: (n, t)

        if not self.shared:
            n_q = codec_tokens.size(0)
            offset = torch.cumsum(
                torch.tensor([0] + [self.num_codebook_per_layer] * (n_q - 1)), 0
            ).unsqueeze(1)
            offset = (
                offset + self.num_special_tokens + self.num_instrument_tokens
            )
        else:
            offset = self.num_special_tokens + self.num_instrument_tokens
        return codec_tokens + offset

    def __call__(self, item):

        target_token = item["target_token"]
        target_token = self.codec_tokens_to_tokens(target_token)
        item["target_token"] = target_token

        # add num_special_tokens to instrument id
        item["target_inst_token"] = (
            self.num_special_tokens + item["target_inst_id"]
        )
        return item


class Serialize(BaseTransform):
    """Serialize RVQ tokens to a sequence."""

    def __call__(self, item):
        # now we do simple flatten
        output_token = item["target_token"]

        # token: [n, t]
        # --> [t,n] --> flatten to [n*t]
        output_token = output_token.t().reshape(-1)

        item["target_token"] = output_token
        return item


class AddSpecialTokens(BaseTransform):
    """
    Add special tokens to the tokens to target tokens.

    Inputs:
        multilayer (bool): flag to indicate when we are loading non-flattened
            RVQ tokens for the target_tokens. Used for delay-patterning.
        add_inst_tokens: flag to indicate if we should add instrument tokens
    """

    def __init__(
        self,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        add_eos: bool = True,
        add_bos: bool = True,
        multilayer: bool = False,
        add_inst_tokens: bool = True,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_eos = add_eos
        self.add_bos = add_bos
        self.multilayer = multilayer
        self.add_inst_tokens = add_inst_tokens

    def __call__(self, item):
        target_token = item["target_token"]
        target_instrument_token = item["target_inst_token"]

        if not self.multilayer:
            target_prefix_tokens = torch.empty(0, dtype=target_token.dtype)
            if self.add_bos:
                pass
                # don't add bos token to target, use instrument id instead
                # target_prefix_tokens = torch.cat(
                #     [target_prefix_tokens, torch.tensor([self.bos_token_id])]
                # )
            # add instrument id, only to target
            if self.add_inst_tokens:
                target_prefix_tokens = torch.cat(
                    [
                        target_prefix_tokens,
                        torch.tensor([target_instrument_token]),
                    ]
                )

            target_token = torch.cat([target_prefix_tokens, target_token])

            if self.add_eos:
                target_token = torch.cat(
                    [target_token, torch.tensor([self.eos_token_id])]
                )
        else:
            if self.add_bos or self.add_eos:
                raise NotImplementedError
            if self.add_inst_tokens:
                raise DeprecationWarning(
                    "delay Pattern Now does NOT concatenate instrument tokens."
                )
                n_rvq = target_token.shape[-2]
                target_prefix_tokens = torch.full(
                    (n_rvq, 1), target_instrument_token
                )
                target_token = torch.cat(
                    (target_prefix_tokens, target_token), dim=-1
                )

        item["target_token"] = target_token

        return item


class SubmixInputEmbEncodec(BaseTransform):
    """Sum variable numbers of Encodec input stems to a single embedding sequence."""

    def __init__(self, codebook):
        self.codebook = codebook

    def codebook_lookup(self, token):
        # token: [n, t]
        codes = token.unsqueeze(1)  # [n, 1, t]
        embeddings = self.codebook.decode(codes)
        # embeddings: [1, d, t]
        embeddings = embeddings.squeeze(0).transpose(0, 1)
        # embeddings: [t, d]
        return embeddings

    def __call__(self, item):
        input_embeddings = [
            self.codebook_lookup(token) for token in item["input_token_stems"]
        ]
        input_embeddings = torch.stack(input_embeddings, dim=0).mean(dim=0)
        item["input_emb"] = input_embeddings
        return item


class SubmixInputEmbDAC(BaseTransform):
    """Sum variable numbers of DAC input stems to a single embedding sequence."""

    def __init__(self, codebook):
        self.codebook = codebook
        self.codebook.eval()

    @torch.no_grad()
    def codebook_lookup(self, token):
        # token: [n, t]
        embeddings, _, _ = self.codebook.from_codes(token[None])  # [1, d, t]
        embeddings = embeddings.squeeze().transpose(0, 1)  # [t, d]
        return embeddings

    def __call__(self, item):
        input_embeddings = [
            self.codebook_lookup(token) for token in item["input_token_stems"]
        ]
        input_embeddings = torch.stack(input_embeddings, dim=0).mean(dim=0)
        item["input_emb"] = input_embeddings
        return item


class PrecomputedInputTokenToEmbDAC(BaseTransform):
    """Convert precomputed input tokens to embeddings."""

    def __init__(self, codebook):
        self.codebook = codebook
        self.codebook.eval()

    @torch.no_grad()
    def codebook_lookup(self, token):
        # token: [n, t]
        embeddings, _, _ = self.codebook.from_codes(token[None])  # [1, d, t]
        embeddings = embeddings.squeeze().transpose(0, 1)  # [t, d]
        return embeddings

    def __call__(self, item):
        input_embeddings = self.codebook_lookup(item["input_token"])
        item["input_emb"] = input_embeddings
        # We don't need to mask the tokens here, because in the dataset dump,
        # the audio are padded with silence, so the tokens are all valid.
        item["input_emb_mask"] = torch.ones(
            item["input_token"].shape[1], dtype=torch.bool
        )
        item["input_token_stems"] = [item["input_token"]]
        return item


class RemoveKeys(BaseTransform):
    """Remove keys from the item."""

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, item):
        for key in self.keys:
            if key in item:
                del item[key]
        return item
