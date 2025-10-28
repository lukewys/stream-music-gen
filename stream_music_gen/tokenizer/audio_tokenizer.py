"""Generic audio tokenizer interface for all audio tokenizers."""

import torch
import torch.nn as nn

from transformers import EncodecModel, AutoProcessor
from audiotools import AudioSignal
from dac import DAC

from stream_music_gen.constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    ENCODEC_FRAME_RATE_HZ,
    ENCODEC_NUM_CODEBOOK_PER_LAYER,
    ENCODEC_SAMPLE_RATE,
    ENCODEC_EMB_DIM,
    DAC_FRAME_RATE_HZ,
    DAC_NUM_CODEBOOK_PER_LAYER,
    DAC_SAMPLE_RATE,
    DAC_PRETRAINED_MODEL_PATH,
    DAC_EMB_DIM,
)


class BaseAudioTokenizer(nn.Module):
    """Base class for audio tokenizers."""

    def __init__(
        self,
        pad_token: int = PAD_TOKEN,
        bos_token: int = BOS_TOKEN,
        eos_token: int = EOS_TOKEN,
    ):
        super().__init__()
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.num_special_tokens = 3

    def audio_to_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to tokens used by language model."""
        raise NotImplementedError("Subclasses must implement this method")

    def audio_to_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to embeddings."""
        raise NotImplementedError("Subclasses must implement this method")

    def tokens_to_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to audio."""
        raise NotImplementedError("Subclasses must implement this method")

    def tokens_to_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to embeddings."""
        raise NotImplementedError("Subclasses must implement this method")

    def embeddings_to_audio(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to audio."""
        raise NotImplementedError("Subclasses must implement this method")

    def embeddings_to_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to tokens used by language model."""
        raise NotImplementedError("Subclasses must implement this method")

    def codec_tokens_to_tokens(
        self, codec_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Convert codec tokens to tokens used by language model."""
        raise NotImplementedError("Subclasses must implement this method")

    def tokens_to_codec_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens used by language model to codec tokens."""
        raise NotImplementedError("Subclasses must implement this method")


class EncodecAudioTokenizer(BaseAudioTokenizer):
    """Audio tokenizer for Encodec model."""

    def __init__(
        self,
        num_special_tokens: int = 3,
        num_instrument_tokens=20,
        num_rvq_layers: int = 4,
    ):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_32khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
        self.num_codebook_per_layer = ENCODEC_NUM_CODEBOOK_PER_LAYER
        self.sample_rate = ENCODEC_SAMPLE_RATE
        self.frame_rate = ENCODEC_FRAME_RATE_HZ
        self.emb_dim = ENCODEC_EMB_DIM
        self.num_special_tokens = num_special_tokens
        self.num_instrument_tokens = num_instrument_tokens
        self.num_non_codec_tokens = (
            self.num_special_tokens + self.num_instrument_tokens
        )
        self.num_rvq_layers = num_rvq_layers
        self.num_tokens = (
            self.num_codebook_per_layer * num_rvq_layers
            + self.num_special_tokens
            + self.num_instrument_tokens
        )
        print(
            f"Num tokens: {self.num_tokens}, "
            f"num rvq layers: {self.num_rvq_layers}, "
            f"num codebook per layer: {self.num_codebook_per_layer}, "
            f"num special tokens: {num_special_tokens}, "
            f"num instrument tokens: {num_instrument_tokens}, "
            f"num non codec tokens: {self.num_non_codec_tokens}, "
            f"frame rate: {self.frame_rate}, "
            f"sample rate: {self.sample_rate}"
        )
        self.model.eval()
        # disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def audio_to_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to tokens used by language model."""
        # audio: (b, t)
        audio = audio.squeeze().numpy()
        # Processor takes in a list of np array for batch processing
        audio = [a for a in audio]
        device = next(self.model.parameters()).device
        inputs = self.processor(
            raw_audio=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        # extract full rvq layers
        bandwidth = self.model.config.target_bandwidths[-1]
        encoder_outputs = self.model.encode(
            inputs["input_values"].to(device),
            inputs["padding_mask"].to(device),
            bandwidth=bandwidth,
        )
        codes = encoder_outputs.audio_codes.squeeze()
        tokens = self.codec_tokens_to_tokens(codes)
        return tokens

    @torch.no_grad()
    def tokens_to_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to audio."""
        codec_tokens = self.tokens_to_codec_tokens(tokens)
        # codec_tokens: (b, n, t)
        if (
            codec_tokens.max() >= self.num_codebook_per_layer
            or codec_tokens.min() < 0
        ):
            raise ValueError(
                f"Invalid tokens: {codec_tokens.min()}, {codec_tokens.max()}, "
                f"correct range: {self.num_codebook_per_layer}"
            )
        return self.codec_tokens_to_audio(codec_tokens)

    @torch.no_grad()
    def codec_tokens_to_audio(self, codec_tokens: torch.Tensor) -> torch.Tensor:
        """Convert codec tokens to audio."""
        # codec_tokens: (b, n, t)
        audio_all = []
        for i in range(codec_tokens.size(0)):
            audio = self.model.decode(
                codec_tokens[i].unsqueeze(0).unsqueeze(0),
                [None],  # scale is None for 24/32khz model
                None,  # don't use mask for now
            )[0]
            audio_all.append(audio)
        audio = torch.stack(audio_all, dim=0).squeeze()  # (t) or (b, t)
        return audio

    @torch.no_grad()
    def codec_tokens_to_tokens(
        self, codec_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Convert codec tokens to tokens used by language model."""
        # codec_tokens: (b, n, t)
        n_q = codec_tokens.size(1)
        offset = (
            torch.cumsum(
                torch.tensor(
                    [0] + [self.num_codebook_per_layer] * (n_q - 1),
                    device=codec_tokens.device,
                ),
                0,
            )
            .unsqueeze(0)
            .unsqueeze(2)
        )
        offset = offset + self.num_non_codec_tokens
        return codec_tokens + offset

    @torch.no_grad()
    def tokens_to_codec_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens used by language model to codec tokens."""
        # tokens: (b, n, t)
        n_q = tokens.size(1)

        offset = (
            torch.cumsum(
                torch.tensor(
                    [0] + [self.num_codebook_per_layer] * (n_q - 1),
                    device=tokens.device,
                ),
                0,
            )
            .unsqueeze(0)
            .unsqueeze(2)
        )  # (1, n, 1)
        offset = offset + self.num_non_codec_tokens
        return tokens - offset

    def post_process_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Post process tokens output from dataloader or LM to codec tokens in [n, t]."""
        # tokens: (T)
        # remove special tokens
        tokens_no_special = tokens[tokens >= self.num_non_codec_tokens]
        # reshape to [n, t]
        codec_tokens = tokens_no_special.reshape(
            -1, self.num_rvq_layers
        ).permute(1, 0)
        return codec_tokens


class DACAudioTokenizer(BaseAudioTokenizer):
    """
    Audio tokenizer for DAC model.

    Attributes:
        multilayer (bool): flag to indicate when we are loading non-flattened RVQ tokens
            for the target_tokens. Used for delay-patterning.
        shared (bool): flag to indicate if the vocabulary is shared between RVQ levels.
    """

    def __init__(
        self,
        num_special_tokens: int = 3,
        num_instrument_tokens: int = 20,
        num_rvq_layers: int = 4,
        multilayer: bool = False,
        shared: bool = False,
    ):
        super().__init__()
        self.model = DAC.load(DAC_PRETRAINED_MODEL_PATH)
        assert self.model.causal_decoder and self.model.causal_encoder
        self.num_codebook_per_layer = DAC_NUM_CODEBOOK_PER_LAYER
        self.sample_rate = DAC_SAMPLE_RATE
        self.frame_rate = DAC_FRAME_RATE_HZ
        self.emb_dim = DAC_EMB_DIM
        self.num_special_tokens = num_special_tokens
        self.num_instrument_tokens = num_instrument_tokens
        self.num_non_codec_tokens = (
            self.num_special_tokens + self.num_instrument_tokens
        )
        self.num_rvq_layers = num_rvq_layers

        self.multilayer = multilayer
        self.shared = shared

        if not shared:
            self.num_tokens = (
                self.num_codebook_per_layer * num_rvq_layers
                + self.num_special_tokens
                + self.num_instrument_tokens
            )
        else:
            self.num_tokens = (
                self.num_codebook_per_layer
                + self.num_special_tokens
                + self.num_instrument_tokens
            )

        print(
            f"Num tokens: {self.num_tokens}, "
            f"num rvq layers: {self.num_rvq_layers}, "
            f"num codebook per layer: {self.num_codebook_per_layer}, "
            f"num special tokens: {num_special_tokens}, "
            f"num instrument tokens: {num_instrument_tokens}, "
            f"num non codec tokens: {self.num_non_codec_tokens}, "
            f"frame rate: {self.frame_rate}, "
            f"sample rate: {self.sample_rate}"
            f"shared: {self.shared}"
        )
        self.model.eval()
        # disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def audio_to_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to tokens used by language model."""
        device = next(self.model.parameters()).device
        # audio: (b, t)
        audio = audio.squeeze()
        win_duration = audio.shape[-1] / self.sample_rate + 1
        audio = [
            AudioSignal(a, sample_rate=self.sample_rate).to(device)
            for a in audio
        ]
        encoder_outputs = self.model.compress(audio, win_duration=win_duration)
        codes = encoder_outputs.codes.squeeze()
        tokens = self.codec_tokens_to_tokens(codes)
        return tokens

    @torch.no_grad()
    def tokens_to_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens to audio."""
        codec_tokens = self.tokens_to_codec_tokens(tokens)
        # codec_tokens: (b, n, t)
        if (
            codec_tokens.max() >= self.num_codebook_per_layer
            or codec_tokens.min() < 0
        ):
            raise ValueError(
                f"Invalid tokens: {codec_tokens.min()}, {codec_tokens.max()}, "
                f"correct range: {self.num_codebook_per_layer}"
            )
        return self.codec_tokens_to_audio(codec_tokens)

    @torch.no_grad()
    def codec_tokens_to_audio(self, codec_tokens: torch.Tensor) -> torch.Tensor:
        """Convert codec tokens to audio."""
        # codec_tokens: (b, n, t)
        device = next(self.model.parameters()).device
        # z_q: (b, d, t)
        z_q, _, _ = self.model.quantizer.from_codes(codec_tokens.to(device))
        # recon: (b, 1, t_audio)
        recon = self.model.decode(z_q)
        start_samp = self.model.hop_length - 1
        num_frames = codec_tokens.shape[2]
        num_samples = int(num_frames / self.frame_rate * self.sample_rate)
        recon = recon[..., start_samp : start_samp + num_samples]
        audio = recon.squeeze()  # (b, t) or (t)
        return audio

    @torch.no_grad()
    def codec_tokens_to_tokens(
        self, codec_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Convert codec tokens to tokens used by language model."""
        # codec_tokens: (b, n, t)
        n_q = codec_tokens.size(1)

        if not self.shared:
            offset = (
                torch.cumsum(
                    torch.tensor(
                        [0] + [self.num_codebook_per_layer] * (n_q - 1),
                        device=codec_tokens.device,
                    ),
                    0,
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )

            offset = offset + self.num_non_codec_tokens
        else:
            offset = self.num_non_codec_tokens
        return codec_tokens + offset

    @torch.no_grad()
    def tokens_to_codec_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens used by language model to codec tokens."""
        # tokens: (b, n, t)
        n_q = tokens.size(1)

        if not self.shared:
            offset = (
                torch.cumsum(
                    torch.tensor(
                        [0] + [self.num_codebook_per_layer] * (n_q - 1),
                        device=tokens.device,
                    ),
                    0,
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )  # (1, n, 1)
            offset = offset + self.num_non_codec_tokens
        else:
            offset = self.num_non_codec_tokens
        return tokens - offset

    def post_process_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Post process tokens output from dataloader or LM to codec tokens in [n, t].
        Args:
            multilayer (bool): flag to indicate when we are loading non-flattened RVQ tokens
                for the target_tokens. Used for delay-patterning.
        """
        # tokens: (T) if not multilayer, [N,T] if they are.

        # remove special tokens
        if not self.multilayer:
            assert tokens.ndim == 1  # Batch-wise not supported
            tokens_no_special = tokens[tokens >= self.num_non_codec_tokens]
            # reshape to [n, t]
            codec_tokens = tokens_no_special.reshape(
                -1, self.num_rvq_layers
            ).permute(1, 0)
        else:
            assert tokens.ndim == 2  # Batch-wise not supported
            min_token_ids = torch.min(tokens, dim=0, keepdim=False)[0]
            tokens_no_special = tokens[
                :, min_token_ids >= self.num_non_codec_tokens
            ]
            codec_tokens = tokens_no_special.reshape(self.num_rvq_layers, -1)

        return codec_tokens
