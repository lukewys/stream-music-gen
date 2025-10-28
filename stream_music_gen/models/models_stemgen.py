"""Stemgen model https://arxiv.org/pdf/2312.08723."""

import torch
import torch.nn as nn
import math
from tqdm import tqdm

from stream_music_gen.nn.transformers import Encoder
from stream_music_gen.models import TransformerWrapperNoInitEmb


def gumbel_noise_like(t):
    noise = torch.zeros_like(t).uniform_(1e-20, 1)
    return -torch.log(-torch.log(noise))


def cosine_schedule(progress):
    """
    Returns a mask ratio based on progress in [0, 1].
    At progress=0, returns 1 (all tokens masked);
    at progress=1, returns 0 (no tokens masked).
    """
    return torch.cos(progress * torch.pi / 2)


def stemgen_mask(x):
    """Apply stemgen masking.

    This is similar to soundstorm and vampnet, which adapt from maskGIT.

    level mask: random level L, different for each batch.

    time mask:
    p = cos(u), u ~ U[0, π/2] --> [0, 1] of shape T

    Don't mask tokens below the L, mask all tokens above the L.
    Mask tokens at level L according to time mask.
    """
    # x: [b, k, t]
    B, K, T = x.shape

    # level mask
    level_mask = torch.randint(0, K, (B,))  # [B]

    # time mask
    u = torch.rand(B)
    p = cosine_schedule(u)  # [B]
    # apply u to each time step
    time_mask = torch.bernoulli(p.unsqueeze(1).repeat(1, T))  # [B, T]

    # apply mask
    mask = torch.zeros_like(x)
    for b in range(B):
        mask[b, level_mask[b], :] = time_mask[b]
        mask[b, : level_mask[b], :] = 0
        mask[b, level_mask[b] + 1 :, :] = 1

    # 1 = masked token, 0 = unmasked token
    return mask.bool()


class StemGenInputEmb(nn.Module):
    """Input embedding layer for stemgen model."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_tokens: int,
        num_rvq_layers: int,
        num_codebook_per_layer: int,
        num_instrument_tokens: int,
    ):
        super().__init__()
        self.model_dim = model_dim  # D
        self.target_token_emb = nn.Embedding(num_tokens, model_dim // 2)
        self.num_rvq_layers = num_rvq_layers  # K
        self.num_codebook_per_layer = num_codebook_per_layer
        self.num_instrument_tokens = num_instrument_tokens
        self.inst_emb = nn.Embedding(num_instrument_tokens, model_dim // 2)
        self.input_fc = nn.Linear(input_dim, model_dim // 2)

    def embed_target_token(self, target_token):
        """Embed target token."""
        # target_token: [B, K, T]
        # cumsum along K dim
        offset = torch.cumsum(
            torch.tensor(
                [0] + [self.num_codebook_per_layer] * (self.num_rvq_layers - 1),
                dtype=torch.int32,
                device=target_token.device,
            ),
            0,
        )  # [K]
        target_token = target_token + offset.unsqueeze(0).unsqueeze(2)

        target_token_emb = self.target_token_emb(
            target_token
        )  # [B, K, T, D//2]
        target_token_emb = target_token_emb.sum(dim=1)  # [B, T, D//2]
        return target_token_emb

    def forward(self, model_input):
        """
        Model input: {
            input: [B, T, D],
            target_masked: [B, K, T],
            target_inst_id: [B],
        }
        """
        input_emb = self.input_fc(model_input["input"])  # [B, T, D//2]
        target_token_emb = self.embed_target_token(
            model_input["target_masked"]
        )  # [B, T, D//2]
        inst_emb = self.inst_emb(model_input["target_inst_id"])  # [B, D//2]

        # concat input_emb, target_token_emb, add inst_emb to each stream
        input_emb = input_emb + inst_emb.unsqueeze(1).repeat(
            1, input_emb.size(1), 1
        )  # [B, T, D]
        target_token_emb = target_token_emb + inst_emb.unsqueeze(1).repeat(
            1, target_token_emb.size(1), 1
        )  # [B, T, D]

        return torch.cat([input_emb, target_token_emb], dim=-1)  # [B, T, D]


class StemGenOutputLayer(nn.Module):
    """Output layer for stemgen model."""

    def __init__(
        self, num_rvq_layers: int, model_dim: int, num_output_tokens: int
    ):
        super().__init__()
        self.num_rvq_layers = num_rvq_layers
        self.model_dim = model_dim
        self.num_output_tokens = num_output_tokens
        self.output_fc = nn.ModuleList(
            [
                nn.Linear(model_dim, num_output_tokens)
                for _ in range(num_rvq_layers)
            ]
        )

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return torch.stack(
            [fc(x) for fc in self.output_fc], dim=1
        )  # [B, K, T, C]


class StemGen(nn.Module):
    """Encoder-only stemgen model."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        attention_layer_configs: dict = dict(),
        transformer_configs: dict = dict(),
        num_special_tokens: int = 2,
        max_seq_len: int = 512,
        num_rvq_layers: int = 4,
        num_codebook_per_layer: int = 16,
        input_emb_dim: int = 128,
        num_instrument_tokens: int = 16,
        mask_id: int = 1,
        num_sample_steps: list[int] = [128, 64, 32, 32],
        **kwargs,
    ):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.num_rvq_layers = num_rvq_layers
        self.num_input_tokens = (
            num_special_tokens + num_codebook_per_layer * num_rvq_layers
        )
        self.num_output_tokens = num_codebook_per_layer
        self.input_emb = StemGenInputEmb(
            input_dim=input_emb_dim,
            model_dim=dim,
            num_tokens=self.num_input_tokens,
            num_rvq_layers=num_rvq_layers,
            num_codebook_per_layer=num_codebook_per_layer,
            num_instrument_tokens=num_instrument_tokens,
        )
        self.output_layer = StemGenOutputLayer(
            num_rvq_layers=num_rvq_layers,
            model_dim=dim,
            num_output_tokens=num_codebook_per_layer,
        )
        self.encoder = TransformerWrapperNoInitEmb(
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=self.num_input_tokens,
            max_seq_len=max_seq_len,
            token_emb=nn.Identity(),  # no token embedding
            to_logits=self.output_layer,
            **transformer_configs,
        )
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.max_seq_len = max_seq_len
        self.mask_id = mask_id
        assert len(num_sample_steps) == self.num_rvq_layers
        self.num_sample_steps = num_sample_steps

    def forward(self, model_input, mask=None):
        """
        Model input: {
            input: [B, T, D],
            target_masked: [B, K, T],
            target_inst_id: [B],
        }
        """
        input_emb = self.input_emb(model_input)
        return self.encoder(input_emb, mask=mask)

    def get_confidence_stemgen(
        self,
        logits: torch.Tensor,
        curr_level: int,
        sample_mask: torch.Tensor,
        target_masked: torch.Tensor,
        generate_kwargs: dict,
    ):
        """
        Get the confidence score.

        logits: [B, K, T, C]
        sample_mask: [B, T]: True means the token is masked
        target_masked: [B, K, T]
        """
        device = logits.device
        B, K, T, C = logits.shape

        # get probabilities
        probs = torch.softmax(
            logits / generate_kwargs["sample_temperature"], dim=-1
        )  # [B, K, T, C]
        # only takes the probs of the current level
        probs = probs[:, curr_level, :, :]  # [B, T, C]

        # sample from model
        # Reshape probs to [B*T, C] for multinomial
        probs_shape = probs.shape
        probs_flat = probs.reshape(-1, probs_shape[-1])  # [B*T, C]
        sampled_tokens_flat = torch.multinomial(
            probs_flat, num_samples=1
        )  # [B*T]
        sampled_tokens = sampled_tokens_flat.reshape(probs_shape[:-1])  # [B, T]

        # First replace already sampled tokens in sampled_tokens with target_masked
        sampled_tokens = torch.where(
            sample_mask,
            sampled_tokens,
            target_masked[:, curr_level, :] - self.num_special_tokens,
        )

        # get probabilities of sampled tokens
        # [B, T]
        sampled_probs = probs.gather(
            dim=-1, index=sampled_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # calucalte confidence
        random_noise = torch.rand_like(sampled_probs)
        causal_bias = 1 - torch.arange(T, device=device).repeat(B, 1) / T
        confidence = (
            generate_kwargs["prob_conf_coeff"] * sampled_probs
            + generate_kwargs["causal_conf_coeff"] * causal_bias
            + generate_kwargs["random_conf_coeff"] * random_noise
        )  # [B, T]

        # for unmasked tokens, set confidence to infinity
        confidence[~sample_mask] = torch.inf
        return confidence, sampled_tokens

    def get_confidence_vampnet(
        self,
        logits: torch.Tensor,
        curr_level: int,
        sample_mask: torch.Tensor,
        target_masked: torch.Tensor,
        generate_kwargs: dict,
        current_step: int,
        total_steps: int,
    ):
        """
        Get the confidence score.

        logits: [B, K, T, C]
        sample_mask: [B, T]: True means the token is masked
        target_masked: [B, K, T]
        current_step: int 1-indexed current sampling step
        total_steps: int total number of sampling steps
        """

        if len(generate_kwargs["noise_temperature"]) != self.num_rvq_layers:
            raise ValueError(
                f"noise_temperature must be of length {self.num_rvq_layers}"
            )

        # get probabilities
        log_probs = torch.log_softmax(
            logits / generate_kwargs["sample_temperature"], dim=-1
        )  # [B, K, T, C]
        # only takes the probs of the current level
        log_probs = log_probs[:, curr_level, :, :]  # [B, T, C]

        # sample from model
        # Reshape probs to [B*T, C] for multinomial
        log_probs_shape = log_probs.shape
        log_probs_flat = log_probs.reshape(-1, log_probs_shape[-1])  # [B*T, C]
        sampled_tokens_flat = torch.multinomial(
            log_probs_flat.exp(), num_samples=1
        )  # [B*T]
        sampled_tokens = sampled_tokens_flat.reshape(
            log_probs_shape[:-1]
        )  # [B, T]

        # First replace already sampled tokens in sampled_tokens with target_masked
        sampled_tokens = torch.where(
            sample_mask,
            sampled_tokens,
            target_masked[:, curr_level, :] - self.num_special_tokens,
        )

        # get probabilities of sampled tokens
        # [B, T]
        sampled_log_probs = log_probs.gather(
            dim=-1, index=sampled_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # calucalte confidence
        random_noise = gumbel_noise_like(sampled_log_probs)
        noise_temperature = generate_kwargs["noise_temperature"][curr_level]
        use_temperature_annealing = generate_kwargs.get(
            "use_temperature_annealing", True
        )
        if use_temperature_annealing:
            noise_temperature = noise_temperature * (
                (total_steps - current_step) / total_steps
            )
        confidence = (
            sampled_log_probs + noise_temperature * random_noise
        )  # [B, T]

        # for unmasked tokens, set confidence to infinity
        confidence[~sample_mask] = torch.inf
        return confidence, sampled_tokens

    @torch.no_grad()
    def generate(
        self,
        input_context,
        target_inst_id,
        mask: torch.Tensor = None,
        display_pbar: bool = False,
        guidance_scale: float = 1.0,
        confidence_method: str = "stemgen",
        generate_kwargs: dict = dict(),
    ):
        # starts with all masked tokens
        device = input_context.device
        B, T, _ = input_context.shape
        K = self.num_rvq_layers
        target_masked = torch.full(
            (B, K, T),
            self.mask_id,
            device=input_context.device,
        )

        for curr_level in range(self.num_rvq_layers):
            if display_pbar:
                pbar = tqdm(
                    range(self.num_sample_steps[curr_level]),
                    desc=f"Layer {curr_level}",
                )
            else:
                pbar = range(self.num_sample_steps[curr_level])

            # create a mask for the current level, True means the token is masked
            sample_mask = torch.ones([B, T], device=device).bool()
            for t in pbar:

                model_input = {
                    "input": input_context,
                    "target_masked": target_masked,
                    "target_inst_id": target_inst_id,
                }
                # run the model
                logits = self.forward(model_input, mask)  # [B, K, T, C]

                if guidance_scale != 1.0:
                    # cfg
                    cfg_model_input = {
                        "input": torch.zeros_like(input_context),
                        "target_masked": target_masked,
                        "target_inst_id": target_inst_id,
                    }
                    cfg_logits = self.forward(
                        cfg_model_input, mask
                    )  # [B, K, T, C]

                    logits = cfg_logits + guidance_scale * (logits - cfg_logits)

                if confidence_method == "stemgen":
                    confidence, sampled_tokens = self.get_confidence_stemgen(
                        logits,
                        curr_level=curr_level,
                        sample_mask=sample_mask,
                        target_masked=target_masked,
                        generate_kwargs=generate_kwargs,
                    )
                elif confidence_method == "vampnet":
                    confidence, sampled_tokens = self.get_confidence_vampnet(
                        logits,
                        curr_level=curr_level,
                        sample_mask=sample_mask,
                        target_masked=target_masked,
                        generate_kwargs=generate_kwargs,
                        current_step=t + 1,  # 1-indexed
                        total_steps=self.num_sample_steps[curr_level],
                    )
                else:
                    raise ValueError(
                        f"Confidence method {confidence_method} not supported"
                    )

                # mask according to confidence
                # progress in [0, 1]
                sample_progress = (
                    torch.tensor(
                        (t + 1) / self.num_sample_steps[curr_level],
                        device=device,
                    )
                    .unsqueeze(0)
                    .repeat(B, 1)
                )  # [B, 1]
                mask_prob = cosine_schedule(sample_progress)
                num_mask = (mask_prob * T).long()

                if t != (self.num_sample_steps[curr_level] - 1):
                    num_mask = torch.maximum(
                        torch.tensor(1),
                        torch.minimum(
                            sample_mask.sum(dim=-1, keepdim=True) - 1, num_mask
                        ),
                    )

                # Top-k masking
                for b in range(B):
                    confidence_rank = torch.argsort(
                        confidence[b], dim=-1
                    )  # [T]
                    # Update mask - True means token will be masked
                    sample_mask[b, confidence_rank[: num_mask[b]]] = True
                    sample_mask[b, confidence_rank[num_mask[b] :]] = False

                # update target_masked, add special tokens offset
                target_masked[:, curr_level, :] = (
                    sampled_tokens + self.num_special_tokens
                )
                # mask target_masked
                target_masked[:, curr_level, :] = target_masked[
                    :, curr_level, :
                ].masked_fill(sample_mask, self.mask_id)

        return target_masked
