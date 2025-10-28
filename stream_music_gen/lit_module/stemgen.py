"""Lightning module for training the stemgen (encoder-only) model."""

import argbind
import time

import wandb
import torch

import torch.nn.functional as F

from stream_music_gen.base_trainer import BaseLightningModel
from stream_music_gen.utils.lr_scheduler import LinearWarmupCosineDecay
from stream_music_gen.utils.plot_utils import audio_to_spectrogram_image
from stream_music_gen.models import StemGen, stemgen_mask
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

StemGen = bind(StemGen)
AdamW = bind(torch.optim.AdamW)
LinearWarmupCosineDecay = bind(LinearWarmupCosineDecay)
get_dataloader = bind(get_precomputed_token_dataloader, without_prefix=True)


@bind(without_prefix=True)
class LitStemGen(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        max_duration: int = 10,
        cfg_drop_prob: float = 0.2,
        # use 1 as mask token, we don't have eos token here
        mask_token_id: int = 1,
        num_rvq_layers_to_use: int = 4,
        generate_kwargs: dict = dict(),
    ):
        super(LitStemGen, self).__init__()

        self.num_special_tokens = 2  # pad and mask token
        self.num_instrument_tokens = len(MIDI_CATEGORIES)
        tokenizer = DACAudioTokenizer(
            num_special_tokens=self.num_special_tokens,
            # No instrument tokens for audio tokenizer in stemgen
            num_instrument_tokens=0,
            num_rvq_layers=num_rvq_layers_to_use,
            multilayer=True,
            shared=True,
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
            pattern="stemgen",
        )
        val_dataloader = get_dataloader(
            frame_rate_hz=self.frame_rate,
            duration=self.max_duration,
            num_rvq_layers=self.num_rvq_layers,
            sample_rate=self.sample_rate,
            num_codebook_per_layer=self.num_codebook_per_layer,
            split="valid",
            shuffle=False,
            pattern="stemgen",
            load_audio="true",
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples

        tokenizer.eval()
        self.pad_token = tokenizer.pad_token
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.tokenizer = self.tokenizer.to(torch.device("cpu"))
        self.sample_rate = tokenizer.sample_rate
        self.cfg_drop_prob = cfg_drop_prob
        self.mask_token_id = mask_token_id

        self.model = StemGen(
            num_special_tokens=2,
            max_seq_len=self.max_gen_seq_len,
            input_emb_dim=tokenizer.emb_dim,
            num_rvq_layers=self.num_rvq_layers,
            num_codebook_per_layer=self.num_codebook_per_layer,
            num_instrument_tokens=len(MIDI_CATEGORIES),
        )

        if compile:
            self.model = torch.compile(self.model)

        self.generate_kwargs = generate_kwargs

    def get_inputs(self, batch):
        enc_inputs = batch["input_emb"]
        targets = batch["target_token"]
        inputs_mask = batch["input_emb_mask"]
        target_inst_id = torch.tensor(
            batch["target_inst_id"], device=enc_inputs.device
        )
        loss_targets = targets - self.num_special_tokens
        return enc_inputs, targets, inputs_mask, target_inst_id, loss_targets

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        # enc_inputs: [B, K, D]
        # targets: [B, K, T]
        # inputs_mask: [B, T], 1 = unmasked, 0 = masked
        # target_inst_id: [B]
        enc_inputs, targets, inputs_mask, target_inst_id, loss_targets = (
            self.get_inputs(batch)
        )

        if self.cfg_drop_prob > 0:
            # for cfg, drop the encoder input and replace them with all 0s
            drop_num = int(enc_inputs.size(0) * self.cfg_drop_prob)
            drop_indices = torch.randperm(enc_inputs.size(0))[:drop_num]
            enc_inputs[drop_indices] = torch.zeros_like(
                enc_inputs[drop_indices]
            )

        sample_mask = stemgen_mask(targets)  # [B, K, T]
        targets_input = targets.masked_fill(sample_mask, self.mask_token_id)

        model_input = {
            "input": enc_inputs,
            "target_masked": targets_input,
            "target_inst_id": target_inst_id,
        }

        output = self.model(
            model_input,
            mask=inputs_mask,
        )  # [B, K, T, C]

        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            loss_targets.reshape(-1),
            reduction="none",
        )  # [B * K * T]

        loss_mask = inputs_mask.unsqueeze(1) * sample_mask  # [B, K, T]
        loss_mask = loss_mask.reshape(-1)  # [B * K * T]
        loss = (loss * loss_mask).sum() / loss_mask.sum()

        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        enc_inputs, targets, inputs_mask, target_inst_id, loss_targets = (
            self.get_inputs(batch)
        )

        sample_mask = stemgen_mask(targets)  # [B, K, T]
        targets_input = targets.masked_fill(sample_mask, self.mask_token_id)

        model_input = {
            "input": enc_inputs,
            "target_masked": targets_input,
            "target_inst_id": target_inst_id,
        }
        output = self.model(
            model_input,
            mask=inputs_mask,
        )  # [B, K, T, C]

        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            loss_targets.reshape(-1),
            reduction="none",
        )  # [B * K * T]

        loss_mask = inputs_mask.unsqueeze(1) * sample_mask  # [B, K, T]
        loss_mask = loss_mask.reshape(-1)  # [B * K * T]

        loss = (loss * loss_mask).sum() / loss_mask.sum()

        # Calculate accuracy
        acc = (output.argmax(dim=-1) == loss_targets).float().mean()

        # Log validation loss (step-based logging)
        self._log_dict({"val/loss": loss, "val/acc": acc})

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(
                batch["input_audio"],
                batch["target_audio"],
                enc_inputs,
                inputs_mask,
                target_inst_id,
                targets,
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
        enc_inputs_mask: torch.Tensor,
        target_inst_id: torch.Tensor,
        targets: torch.Tensor,
        num_stems: torch.Tensor,
    ) -> None:
        # save memory
        torch.cuda.empty_cache()
        self.tokenizer = self.tokenizer.to(enc_inputs.device)

        curr_time = time.time()
        model_pred = self.model.generate(
            enc_inputs,
            target_inst_id,
            mask=enc_inputs_mask,
            guidance_scale=2.0,
            confidence_method="vampnet",
            generate_kwargs=self.generate_kwargs,
        )
        print(
            f"Time to generate: {time.time() - curr_time}, "
            f"shape: {model_pred.shape}"
        )

        # We don't log audio and image into table because there will be bugs
        table_text = wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        for i in range(min(self.max_log_examples, model_pred.size(0))):
            array_gt_text = str(targets[i].cpu().numpy())
            array_gen_text = str(model_pred[i].cpu().numpy())
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
                model_pred[i]
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
