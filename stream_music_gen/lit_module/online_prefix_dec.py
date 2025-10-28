"""
Lightning module for training the encoder-decoder generative
with delay pattern
"""

import argbind
import time

import wandb
import torch
import torch.nn.functional as F
import numpy as np

from stream_music_gen.base_trainer import BaseLightningModel
from stream_music_gen.utils.lr_scheduler import LinearWarmupCosineDecay
from stream_music_gen.utils.plot_utils import audio_to_spectrogram_image
from stream_music_gen.models.models_multi_out import (
    OnlinePrefixDecoderTransformerMultiOut,
)
from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.tokenizer import DACAudioTokenizer
from stream_music_gen.constants import MIDI_CATEGORIES
from stream_music_gen.utils.audio_utils import (
    mix_with_generated_stem,
    PRED_DB_OFFSET_LOUD,
    loudness_normalize_audio,
)

bind = argbind.bind

OnlinePrefixDecoderTransformerMultiOut = bind(
    OnlinePrefixDecoderTransformerMultiOut
)
AdamW = bind(torch.optim.AdamW)
LinearWarmupCosineDecay = bind(LinearWarmupCosineDecay)
get_dataloader = bind(get_precomputed_token_dataloader, without_prefix=True)


@bind(without_prefix=True)
class LitOnlinePrefixDecoderMultiOut(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        max_duration: int = 10,
    ):
        """
        Lightning Module for Online Prefix Decoder, multilayer/multiout.

        Args:
            chunk_length: the length of each chunk, in frames
            chunk_start_prob: In preparing the input, the probability of the input
            as a new chunk (no input context).

        Attributes:
            inst_tokens_as_pattern_token: if we are adding inst_tokens to
            inputs/targets
        """
        super(LitOnlinePrefixDecoderMultiOut, self).__init__()

        tokenizer = DACAudioTokenizer(
            num_special_tokens=1,  # Changed from 3 to 1, no BOS
            num_instrument_tokens=len(MIDI_CATEGORIES),
            num_rvq_layers=4,
            multilayer=True,
            shared=True,
        )

        self.max_duration = max_duration
        self.frame_rate = tokenizer.frame_rate
        self.sample_rate = tokenizer.sample_rate
        self.num_rvq_layers = tokenizer.num_rvq_layers
        self.num_codebook_per_layer = tokenizer.num_codebook_per_layer
        self.max_gen_seq_len = (
            self.max_duration * self.frame_rate  # Modified, not mult. by n_rvq
        )

        # hard-code to not add (concat) inst tokens to output token
        add_inst_tokens = False

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples

        tokenizer.eval()
        self.pad_token = tokenizer.pad_token
        self.num_tokens = tokenizer.num_tokens
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.tokenizer = self.tokenizer.to(torch.device("cpu"))
        self.sample_rate = tokenizer.sample_rate

        self.model = OnlinePrefixDecoderTransformerMultiOut(
            num_tokens=self.num_tokens,  # removed +2, no bos/eos
            max_seq_len=self.max_gen_seq_len + add_inst_tokens + 1,
            pad_value=self.pad_token,
            num_rvq_layers=self.num_rvq_layers,  # Added
            shared=True,
            online=True,  # Added
            inst_tokens_as_pattern_token=True,
            input_emb_dim=tokenizer.emb_dim,
        )

        if compile:
            self.model = torch.compile(self.model)

        # Create dataloaders
        duration = self.max_duration - min(
            0, self.model.input_delay / self.frame_rate
        )

        train_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="train",
            shuffle=True,
            pattern="multilayer",
            add_inst_tokens=add_inst_tokens,  # Added
        )
        val_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="valid",
            shuffle=False,
            pattern="multilayer",
            add_inst_tokens=add_inst_tokens,  # Added
            load_audio="true",
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def get_inputs(self, batch):
        # We will chunk and crop input inside the model class
        input_emb = batch["input_emb"]
        output_tokens = batch["target_token"]

        dec_inst_tokens = torch.tensor(
            batch["target_inst_token"], device=output_tokens.device
        )

        return (
            input_emb,
            output_tokens,
            dec_inst_tokens,
        )

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        # print(f"batch_idx: {batch_idx}, global_step: {self.global_step}")
        input_emb, output_tokens, dec_inst_tokens = self.get_inputs(batch)

        # In x-transformers, mask is True for unmasked tokens
        logits, logits_mask, targets = self.model(
            x=output_tokens,
            input_emb=input_emb,
            inst_tokens=dec_inst_tokens,
        )

        loss = F.cross_entropy(
            logits[logits_mask],  #  For delay pattern, only
            targets[logits_mask],  # compute mask on valid logits.
            ignore_index=self.pad_token,
        )
        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss}, batch_size=output_tokens.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        input_emb, output_tokens, dec_inst_tokens = self.get_inputs(batch)
        logits, logits_mask, targets = self.model(
            x=output_tokens,
            input_emb=input_emb,
            inst_tokens=dec_inst_tokens,
        )
        loss = F.cross_entropy(
            logits[logits_mask],
            targets[logits_mask],
            ignore_index=self.pad_token,
        )

        # Calculate accuracy
        mask = targets != self.pad_token
        acc = (logits.argmax(dim=-1)[mask] == targets[mask]).float().mean()

        # Log validation loss (step-based logging)
        self._log_dict(
            {"val/loss": loss, "val/acc": acc}, batch_size=output_tokens.size(0)
        )

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(
                batch["input_audio"],
                batch["target_audio"],
                input_emb,
                output_tokens,
                dec_inst_tokens,
                batch["num_stems"],
            )

        return loss

    def log_audio(self, audio, name):
        if isinstance(audio, torch.Tensor):
            audio = audio.float().cpu().numpy()
        spectrogram = audio_to_spectrogram_image(audio, self.sample_rate)
        self.logger.experiment.log(
            {
                f"audio/{name}": wandb.Audio(
                    audio, sample_rate=self.sample_rate
                ),
                f"image/{name}": wandb.Image(spectrogram),
            }
        )

    def sample_and_log(
        self,
        input_audios: torch.Tensor,
        target_audios: torch.Tensor,
        input_emb: torch.Tensor,
        dec_inputs: torch.Tensor,
        dec_inst_tokens: torch.Tensor,
        num_stems: torch.Tensor,
    ) -> None:
        # save memory
        torch.cuda.empty_cache()
        self.tokenizer = self.tokenizer.to(input_emb.device)

        # hard-code to use inst tokens
        print("Generating using Instrument Tokens:")
        print(dec_inst_tokens)
        curr_time = time.time()
        decoder_preds = self.model.generate(
            seq_len=self.max_gen_seq_len,
            seq_out_start=None,
            input_emb=input_emb,
            inst_tokens=dec_inst_tokens,
            cache_kv=True,
            filter_logits_fn=["top_k_multi_out"],  # Modified,
            filter_kwargs=[
                {"k": 200},
            ],
        )

        print(
            f"Time to generate: {time.time() - curr_time}, "
            f"shape: {decoder_preds.shape}"
        )

        # We don't log audio and image into table because there will be bugs
        table_text = wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        for i in range(min(self.max_log_examples, decoder_preds.size(0))):
            array_gt_text = str(dec_inputs[i].cpu().numpy())
            array_gen_text = str(decoder_preds[i].cpu().numpy())
            table_text.add_data(i, array_gt_text, array_gen_text)

            input_audio = input_audios[i].cpu()
            target_audio = target_audios[i].cpu()
            num_input_stems = num_stems[i]

            input_audio = input_audio[: self.max_duration * self.sample_rate]
            target_audio = target_audio[: self.max_duration * self.sample_rate]

            if self.global_step == 0:
                mixed_audio_loud = mix_with_generated_stem(
                    input_audio.float().numpy(),
                    target_audio.float().numpy(),
                    num_input_stems,
                    self.sample_rate,
                    pred_db_offset=PRED_DB_OFFSET_LOUD,
                )
                self.log_audio(mixed_audio_loud, f"mixed_gt_loud_{i}")
                mixed_audio = mix_with_generated_stem(
                    input_audio.float().numpy(),
                    target_audio.float().numpy(),
                    num_input_stems,
                    self.sample_rate,
                )
                self.log_audio(mixed_audio, f"mixed_gt_{i}")

                # Only log ground truth audio for once
                self.log_audio(
                    loudness_normalize_audio(
                        input_audio.float().numpy(), self.sample_rate
                    ),
                    f"input_{i}",
                )
                self.log_audio(
                    loudness_normalize_audio(
                        target_audio.float().numpy(), self.sample_rate
                    ),
                    f"target_{i}",
                )

            pred_audio_tokens = self.tokenizer.post_process_tokens(
                decoder_preds[i]
            )
            pred_audio_single = self.tokenizer.tokens_to_audio(
                pred_audio_tokens.unsqueeze(0)
            ).cpu()
            if pred_audio_single.shape[0] < target_audio.shape[0]:
                pred_audio_single = F.pad(
                    pred_audio_single,
                    (0, target_audio.shape[0] - pred_audio_single.shape[0]),
                )
            self.log_audio(
                loudness_normalize_audio(
                    pred_audio_single.float().numpy(), self.sample_rate
                ),
                f"pred_{i}",
            )

            mixed_audio_loud = mix_with_generated_stem(
                input_audio.float().numpy(),
                pred_audio_single.float().numpy(),
                num_input_stems,
                self.sample_rate,
                pred_db_offset=PRED_DB_OFFSET_LOUD,
            )
            self.log_audio(mixed_audio_loud, f"mixed_pred_loud_{i}")
            mixed_audio = mix_with_generated_stem(
                input_audio.float().numpy(),
                pred_audio_single.float().numpy(),
                num_input_stems,
                self.sample_rate,
            )
            self.log_audio(mixed_audio, f"mixed_pred_{i}")

        self.logger.experiment.log({"array_text": table_text})

        # Free up GPU memory
        torch.cuda.empty_cache()
        self.tokenizer = self.tokenizer.to(torch.device("cpu"))

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        scheduler = LinearWarmupCosineDecay(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
