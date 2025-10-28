"""
Lightning module for training the encoder-decoder generative
with delay pattern
"""

import argbind
import time

import wandb
import torch

import torch.nn.functional as F

from stream_music_gen.base_trainer import BaseLightningModel
from stream_music_gen.utils.lr_scheduler import LinearWarmupCosineDecay
from stream_music_gen.utils.plot_utils import audio_to_spectrogram_image
from stream_music_gen.models.models_multi_out import (
    PrefixDecoderTransformerMultiOut,
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

PrefixDecoderTransformerMultiOut = bind(PrefixDecoderTransformerMultiOut)
AdamW = bind(torch.optim.AdamW)
LinearWarmupCosineDecay = bind(LinearWarmupCosineDecay)
get_dataloader = bind(get_precomputed_token_dataloader, without_prefix=True)


@bind(without_prefix=True)
class LitPrefixDecoderMultiOut(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        max_duration: int = 10,
        cfg_drop_prob: float = 0.0,
        inst_tokens_as_pattern_token: bool = True,
    ):
        """
        Lightning Module for Prefix Decoder, multilayer/multiout.

        Attributes:
            inst_tokens_as_pattern_token: if we are adding inst_tokens to
            inputs/targets
        """
        super(LitPrefixDecoderMultiOut, self).__init__()

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

        # Assume inst token was added if inst_tokens are not treated as pattern
        add_inst_tokens = not inst_tokens_as_pattern_token
        self.inst_tokens_as_pattern_token = inst_tokens_as_pattern_token

        # Create dataloaders
        train_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=self.max_duration,
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
            duration=self.max_duration,
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

        print("\nTraining Dataloader Length:")
        print(len(train_dataloader))

        print("\nValid Dataloader Length:")
        print(len(val_dataloader))

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples

        tokenizer.eval()
        self.pad_token = tokenizer.pad_token
        self.num_tokens = tokenizer.num_tokens
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.sample_rate = tokenizer.sample_rate
        self.cfg_drop_prob = cfg_drop_prob

        self.model = PrefixDecoderTransformerMultiOut(
            num_tokens=self.num_tokens,  # removed +2, no bos/eos
            # * 2 because we are using the prefix decoder
            max_seq_len=self.max_gen_seq_len * 2 + add_inst_tokens + 1,
            pad_value=self.pad_token,
            num_rvq_layers=self.num_rvq_layers,  # Added
            shared=True,
            online=False,
            inst_tokens_as_pattern_token=inst_tokens_as_pattern_token,
            input_emb_dim=tokenizer.emb_dim,
        )

        if compile:
            self.model = torch.compile(self.model)

    def get_inputs(self, batch):
        enc_inputs = batch["input_emb"]
        dec_inputs = batch["target_token"]
        targets = batch["target_token"]
        enc_inputs_mask = batch["input_emb_mask"]
        dec_inputs_mask = batch["target_token_mask"]

        true_padding = torch.ones(
            dec_inputs_mask.size(0),
            1,
            device=dec_inputs_mask.device,
            dtype=torch.bool,
        )

        dec_inputs_mask = torch.cat((true_padding, dec_inputs_mask), dim=1)

        dec_inst_tokens = torch.tensor(
            batch["target_inst_token"], device=dec_inputs.device
        )

        return (
            enc_inputs,
            dec_inputs,
            targets,
            enc_inputs_mask,
            dec_inputs_mask,
            dec_inst_tokens,
        )

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        # print(f"batch_idx: {batch_idx}, global_step: {self.global_step}")
        (
            enc_inputs,
            dec_inputs,
            targets,
            enc_inputs_mask,
            dec_inputs_mask,
            dec_inst_tokens,
        ) = self.get_inputs(batch)

        if self.cfg_drop_prob > 0:
            # for cfg, drop the encoder input and replace them with all 0s
            drop_num = int(enc_inputs.size(0) * self.cfg_drop_prob)
            drop_indices = torch.randperm(enc_inputs.size(0))[:drop_num]
            enc_inputs[drop_indices] = torch.zeros_like(
                enc_inputs[drop_indices]
            )

        # In x-transformers, mask is True for unmasked tokens
        logits, logits_mask = self.model(
            dec_inputs,
            input_emb=enc_inputs,
            inst_tokens=dec_inst_tokens,
        )

        loss = F.cross_entropy(
            logits[logits_mask],  #  For delay pattern, only
            targets[logits_mask],  # compute mask on valid logits.
            ignore_index=self.pad_token,
        )
        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss}, batch_size=dec_inputs.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        (
            enc_inputs,
            dec_inputs,
            targets,
            enc_inputs_mask,
            dec_inputs_mask,
            dec_inst_tokens,
        ) = self.get_inputs(batch)

        logits, logits_mask = self.model(
            dec_inputs,
            input_emb=enc_inputs,
            inst_tokens=dec_inst_tokens,
        )
        loss = F.cross_entropy(
            logits[logits_mask],
            targets[logits_mask],
            ignore_index=self.pad_token,
        )

        # Calculate accuracy
        acc = (logits.argmax(dim=-1) == targets).float().mean()

        # Log validation loss (step-based logging)
        self._log_dict(
            {"val/loss": loss, "val/acc": acc}, batch_size=dec_inputs.size(0)
        )

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(
                batch["input_audio"],
                batch["target_audio"],
                enc_inputs,
                dec_inputs,
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
        enc_inputs: torch.Tensor,
        dec_inputs: torch.Tensor,
        dec_inst_tokens: torch.Tensor,
        num_stems: torch.Tensor,
    ) -> None:
        print("Generating using Prefix Decoder:")
        curr_time = time.time()
        decoder_preds = self.model.generate(
            seq_len=self.max_gen_seq_len,
            seq_out_start=dec_inputs,
            input_emb=enc_inputs,
            inst_tokens=dec_inst_tokens,
            cache_kv=False,
            filter_logits_fn=["top_k_multi_out"],  # Modified,
            filter_kwargs=[
                {"k": 200},
            ],
        )

        print(
            f"Time to generate: {time.time() - curr_time}, shape: {decoder_preds.shape}"
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
                        input_audio.float().numpy(),
                        self.sample_rate,
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
                decoder_preds[i],
            )
            pred_audio_single = self.tokenizer.tokens_to_audio(
                pred_audio_tokens.unsqueeze(0)
            ).cpu()

            # Zero-pad if the generation is too short
            if pred_audio_single.shape[-1] < input_audio.shape[-1]:
                pred_audio_single = torch.nn.functional.pad(
                    pred_audio_single,
                    (
                        0,
                        input_audio.shape[-1] - pred_audio_single.shape[-1],
                    ),
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

    def get_dataloaders(self):
        return self.train_dataloader, self.val_dataloader
