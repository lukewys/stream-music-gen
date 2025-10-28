"""Lightning module for training the encoder-decoder generative model."""

import argbind
import time

import wandb
import torch

import torch.nn.functional as F

from stream_music_gen.base_trainer import BaseLightningModel
from stream_music_gen.utils.lr_scheduler import LinearWarmupCosineDecay
from stream_music_gen.utils.plot_utils import audio_to_spectrogram_image
from stream_music_gen.models import DecoderTransformer
from stream_music_gen.dataset.token_dataset import (
    get_precomputed_token_dataloader,
)
from stream_music_gen.tokenizer import DACAudioTokenizer
from stream_music_gen.constants import MIDI_CATEGORIES
from stream_music_gen.utils.audio_utils import (
    loudness_normalize_audio,
)

bind = argbind.bind

DecoderTransformer = bind(DecoderTransformer)
AdamW = bind(torch.optim.AdamW)
LinearWarmupCosineDecay = bind(LinearWarmupCosineDecay)
get_dataloader = bind(get_precomputed_token_dataloader, without_prefix=True)


@bind(without_prefix=True)
class LitDecoder(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        max_duration: int = 10,
    ):
        super(LitDecoder, self).__init__()

        tokenizer = DACAudioTokenizer(
            num_special_tokens=3,
            num_instrument_tokens=len(MIDI_CATEGORIES),
            num_rvq_layers=4,
        )

        self.max_duration = max_duration
        self.frame_rate = tokenizer.frame_rate
        self.sample_rate = tokenizer.sample_rate
        self.num_rvq_layers = tokenizer.num_rvq_layers
        self.num_codebook_per_layer = tokenizer.num_codebook_per_layer
        self.max_gen_seq_len = (
            self.max_duration * self.frame_rate * self.num_rvq_layers
        )

        # Create dataloaders
        train_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=self.max_duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="train",
            shuffle=True,
        )
        val_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=self.max_duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="valid",
            shuffle=False,
            load_audio="true",
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples

        tokenizer.eval()
        self.pad_token = tokenizer.pad_token
        self.num_tokens = tokenizer.num_tokens
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.sample_rate = tokenizer.sample_rate

        self.model = DecoderTransformer(
            num_tokens=self.num_tokens,
            # +2 for bos and eos tokens
            max_seq_len=self.max_gen_seq_len + 2,
            pad_value=self.pad_token,
        )

        if compile:
            self.model = torch.compile(self.model)

    def get_inputs(self, batch):
        dec_inputs = batch["target_token"][:, :-1]
        targets = batch["target_token"][:, 1:]
        dec_inputs_mask = batch["target_token_mask"][:, :-1]
        return dec_inputs, targets, dec_inputs_mask

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        dec_inputs, targets, dec_inputs_mask = self.get_inputs(batch)
        # In x-transformers, mask is True for unmasked tokens
        output = self.model(
            dec_inputs,
            mask=dec_inputs_mask,
        )

        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )
        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        dec_inputs, targets, dec_inputs_mask = self.get_inputs(batch)
        output = self.model(
            dec_inputs,
            mask=dec_inputs_mask,
        )

        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )

        # Calculate accuracy
        acc = (output.argmax(dim=-1) == targets).float().mean()

        # Log validation loss (step-based logging)
        self._log_dict({"val/loss": loss, "val/acc": acc})

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(
                batch["input_audio"],
                batch["target_audio"],
                targets,
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
        targets: torch.Tensor,
    ) -> None:
        gen_inputs = targets[:, 0:1]
        curr_time = time.time()
        decoder_preds = self.model.generate(
            gen_inputs,
            seq_len=self.max_gen_seq_len,
            cache_kv=True,
            eos_token=self.tokenizer.eos_token,
            filter_logits_fn=["rvq_mask_logits", "top_k"],
            filter_kwargs=[
                {
                    "num_non_rvq_tokens": self.tokenizer.num_non_codec_tokens,
                    "num_codebook_per_layer": self.tokenizer.num_codebook_per_layer,
                    "num_rvq_layers": self.tokenizer.num_rvq_layers,
                },
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
            array_gt_text = str(targets[i].cpu().numpy())
            array_gen_text = str(decoder_preds[i].cpu().numpy())
            table_text.add_data(i, array_gt_text, array_gen_text)

            input_audio = input_audios[i].cpu()
            target_audio = target_audios[i].cpu()

            input_audio = loudness_normalize_audio(
                input_audio.float().numpy(), self.sample_rate
            )
            target_audio = loudness_normalize_audio(
                target_audio.float().numpy(), self.sample_rate
            )
            self.log_audio(input_audio, f"input_{i}")
            self.log_audio(target_audio, f"target_{i}")

            pred_audio_tokens = self.tokenizer.post_process_tokens(
                decoder_preds[i]
            )
            pred_audio_single = self.tokenizer.tokens_to_audio(
                pred_audio_tokens.unsqueeze(0)
            ).cpu()
            self.log_audio(
                loudness_normalize_audio(
                    pred_audio_single.float().numpy(), self.sample_rate
                ),
                f"pred_{i}",
            )

        self.logger.experiment.log({"array_text": table_text})

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
