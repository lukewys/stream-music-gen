"""Models for online stem gen. with multiple tokens per output step"""

from typing import Tuple, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from x_transformers import TransformerWrapper, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    join,
)
from tqdm import tqdm

from stream_music_gen.nn.transformers import Decoder, Encoder
from stream_music_gen.models.sampling import (
    top_k_multi_out,
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    validate_filter_fn_kwargs,
)

# from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from stream_music_gen.models.patterns import DelayedPatternProvider
from stream_music_gen.models.models_transformer import (
    TransformerWrapperNoInitEmb,
)


from x_transformers.x_transformers import ScaledSinusoidalEmbedding


class BaseGenerationMixin:
    """
    Base mixin class providing common generation functionality for multi-RVQ transformer models.

    This mixin contains shared logic for parameter validation, pattern processing,
    sampling, and reconstruction that can be used by different transformer architectures.
    """

    def _validate_generation_params(
        self,
        inst_tokens: Optional[torch.Tensor],
        seq_out_start: Optional[torch.Tensor],
        input_emb: Optional[torch.Tensor],
        guidance_scale: float = 1.0,
    ) -> Tuple[int, torch.device]:
        """Validate generation parameters and return batch_size and device."""
        # Model still requires some conditioning
        assert inst_tokens is not None or seq_out_start is not None
        assert (
            input_emb is not None
        ), "input_emb must be provided for generation"

        batch_size = (
            inst_tokens.shape[0]
            if inst_tokens is not None
            else seq_out_start.shape[0]
        )

        assert (
            input_emb.shape[0] == batch_size
        ), f"Batch size mismatch: input_emb has {input_emb.shape[0]} but expected {batch_size}"

        # if using CFG, we expect the batch to be doubled
        if guidance_scale != 1.0:
            assert batch_size % 2 == 0, (
                "For classifier-free guidance, prompts (and corresponding context) "
                "must have an even batch size (first half conditioned, second half null)."
            )

        device = next(self.parameters()).device
        return batch_size, device

    def _process_filter_functions(
        self, filter_logits_fn, filter_kwargs
    ) -> Tuple[Callable, dict]:
        """Process and validate filter functions."""
        # Process filter logits function
        if isinstance(filter_logits_fn, str):
            assert (
                filter_logits_fn in FILTER_LOGITS_FN
            ), f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        filter_fns_is_list = validate_filter_fn_kwargs(
            filter_logits_fn, filter_kwargs
        )
        if filter_fns_is_list:
            filter_fns = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_logits_fn = filter_fns
            filter_kwargs = dict()

        return filter_logits_fn, filter_kwargs

    def _initialize_generation_pattern(
        self,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        seq_out_start: Optional[torch.Tensor],
        inst_tokens: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        int,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Initialize pattern and output sequence for generation."""
        pattern = self.pattern_provider.get_pattern(seq_len)

        # If prompt is not None, keep overwriting current generation with it.
        if seq_out_start is not None:
            prompt_length = seq_out_start.shape[-1]
            patterned_seq_out, _, seq_out_mask = pattern.build_pattern_sequence(
                seq_out_start,
                special_token=self.temp_token,
                keep_only_valid_steps=True,
            )

            # Use inst_tokens as pattern tokens in patterned prompts
            patterned_seq_out = self._replace_temp_token_with_inst_tokens(
                patterned_seq_out, inst_tokens
            )

            # Make out as prompt. +1 for BOS (instrument token)
            out = patterned_seq_out[:, :, : prompt_length + 1]
        else:
            # Use inst_tokens to prompt output
            out = inst_tokens.view(batch_size, 1, 1).expand(
                batch_size, self.num_rvq_layers, 1
            )
            # In here we do not count the BOS token (instrument token) to prompt length
            prompt_length = 0
            patterned_seq_out = None
            seq_out_mask = None

        # Mask needed later to remove invalid outputs after depatterning
        _, _, post_mask = pattern.build_pattern_sequence(
            torch.ones(batch_size, self.num_rvq_layers, seq_len, device=device),
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )

        return out, prompt_length, post_mask, patterned_seq_out, seq_out_mask

    def _sample_next_tokens(
        self,
        logits: torch.Tensor,
        temperature: float,
        greedy: bool,
        filter_logits_fn: Callable,
        filter_kwargs: dict,
        curr_sample_step: int,
        curr_sample_length: int,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample next tokens from logits."""
        # Apply classifier-free guidance if enabled
        if guidance_scale != 1.0:
            # Expecting doubled batch: split into conditioned and null (unconditional)
            half = logits.shape[0] // 2
            logits_cond = logits[:half]
            logits_uncond = logits[half:]
            # Combine the logits using the CFG formula:
            logits = logits_uncond + guidance_scale * (
                logits_cond - logits_uncond
            )

        if greedy:
            samples = logits.argmax(dim=-1, keepdim=True)
        else:
            filtered_logits = filter_logits_fn(
                logits,
                curr_sample_step=curr_sample_step,
                curr_sample_length=curr_sample_length,
                **filter_kwargs,
            )
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            # torch.multinomial cannot handle extra batch dimensions...
            orig_shape = probs.shape[:-1]
            probs_flat = probs.view(-1, self.num_tokens)
            samples_flat = torch.multinomial(probs_flat, 1)
            samples = samples_flat.view(*orig_shape, 1)

        # If using CFG, duplicate the sampled token for both halves
        if guidance_scale != 1.0:
            samples = torch.cat([samples, samples], dim=0)

        return samples

    def _apply_pattern_constraints(
        self,
        out: torch.Tensor,
        curr_sample_step: int,
        seq_out_start: Optional[torch.Tensor],
        patterned_seq_out: Optional[torch.Tensor],
        seq_out_mask: Optional[torch.Tensor],
        post_mask: torch.Tensor,
        inst_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply pattern-based constraints and token replacement."""
        # +1 because we start with 1 token (bos / instrument token)
        curr_step = curr_sample_step + 1

        # Overwrite current output with prompt
        if seq_out_start is not None:
            out[..., curr_step][:, seq_out_mask[..., curr_step]] = (
                patterned_seq_out[..., curr_step][
                    :, seq_out_mask[..., curr_step]
                ]
            )
        else:  # No prompt, so we need to infill instrument token
            out[:, ~post_mask[..., curr_step], curr_step] = self.temp_token
            # Replace pattern token with instrument id
            out = self._replace_temp_token_with_inst_tokens(out, inst_tokens)

        return out

    def _finalize_generation(
        self, out: torch.Tensor, guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """Finalize generation by reverting pattern sequence."""
        pattern = self.pattern_provider.get_pattern(
            out.shape[-1] - 1
        )  # -1 for BOS token

        # Turn generation back into original shape
        reconstructed, rev_indexes, rev_mask = pattern.revert_pattern_sequence(
            out,
            special_token=self.pad_value,
        )

        # If using CFG, return only the conditioned half.
        if guidance_scale != 1.0:
            half = reconstructed.shape[0] // 2
            reconstructed = reconstructed[:half]

        return reconstructed


def generate_positions(length: int, delay: int, device: torch.device):
    """
    Generates input and output position tensors based on the specified length and delay.

    Args:
        length (int): The length of the sequence.
        delay (int): The delay to apply. Positive values shift the input positions backward,
                     negative values shift the output positions backward.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and output position tensors.
    """
    if delay == 0:
        input_pos = torch.arange(length, dtype=torch.long, device=device)
        output_pos = torch.arange(length, dtype=torch.long, device=device)
    elif delay > 0:
        # Calculate the number of valid positions after applying delay
        valid_length = max(length - delay, 0)
        # Create input positions with -1 padding for delay
        input_pos = (
            torch.cat(
                [
                    torch.full((delay,), -1, dtype=torch.long, device=device),
                    torch.arange(valid_length, dtype=torch.long, device=device),
                ]
            )
            if valid_length > 0
            else torch.full((length,), -1, dtype=torch.long, device=device)
        )
        output_pos = torch.arange(length, dtype=torch.long, device=device)
    else:
        # Calculate the number of valid positions after applying negative delay
        valid_length = max(length + delay, 0)
        input_pos = torch.arange(length, dtype=torch.long, device=device)
        output_pos = (
            torch.cat(
                [
                    torch.full((-delay,), -1, dtype=torch.long, device=device),
                    torch.arange(valid_length, dtype=torch.long, device=device),
                ]
            )
            if valid_length > 0
            else torch.full((length,), -1, dtype=torch.long, device=device)
        )
    return input_pos, output_pos


class DelayPatternEmbedder(nn.Module):
    """
    Embeds and sums tokens from multiple RVQ layers, like Audiocraft.

    Attributes:
        dim (int): Embedding dimension
        num_tokens (int): Number of unique tokens in vocabulary
        shared: bool - if the vocabulary is shared between
            RVQ layers in the input dataset.

    Args:
        torch.Tensor (B, K, S):
            B = batch size
            K = number of RVQ layers
            S = sequence length (after any patterning)

    Returns:
        torch.Tensor (B, S, dim)
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        num_rvq_layers: int,
        shared: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.shared = shared
        self.num_rvq_layers = num_rvq_layers
        self.num_tokens = num_tokens

        if not self.shared:
            # Need another for pattern audioLM token.
            self.emb = nn.Embedding(num_tokens, dim)
        else:
            self.emb = nn.Embedding((num_tokens) * num_rvq_layers, dim)

        # Initialize the embeddings
        nn.init.kaiming_normal_(self.emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, S] from delayed pattern
        # Apply offsets, so each codebook gets a unique set of embeddings
        if self.shared:
            offset = (
                torch.cumsum(
                    torch.tensor(
                        [0] + [self.num_tokens] * (self.num_rvq_layers - 1),
                        device=x.device,
                    ),
                    0,
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )
            x = x + offset
        else:
            raise DeprecationWarning(
                "Assume Codebooks Share the Same Vocabulary Ranges"
            )

        x = self.emb(x)  # [B, K, S, D]
        x = x.sum(dim=-3)  # [B, S, D]
        return x


class MultiOutToLogits(nn.Module):
    """
    Parallel output heads for multi-RVQ layer prediction.
    Creates separate linear projections for each RVQ layer.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers/output heads
        dim (int): Input dimension
        num_tokens (int): Output dimension (vocab size)

    Args:
        Input: (B, S, dim)

    Returns:
        Output: (B, K, S, num_tokens), K = num_rvq_layers
    """

    def __init__(self, num_rvq_layers: int, dim: int, num_tokens: int):
        super().__init__()
        self.out_heads = nn.ModuleList(
            [nn.Linear(dim, num_tokens) for _ in range(num_rvq_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(x) for head in self.out_heads], dim=1)


class DecoderTransformerMultiOut(AutoregressiveWrapper, BaseGenerationMixin):
    """
    Transform Decoder with support for multiple RVQ outputs and delay patterns
    Implements staggered generation strategy for parallel RVQ layer prediction.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers to predict
        shared (bool): if the token ids in each RVQ level have the same range.

        online (bool): If True, the model is in online mode and takes in concatenated
            input and output embeddings.
        future_visibility (int): Number of tokens to delay the input stream.
        input_emb_dim (int): Dimension of the input stream embeddings before concatenation.
        output_emb_dim (int): Dimension of the output stream embeddings before concatenation.

        Other args identical to models.py
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,  # Modified
        online: bool = False,
        future_visibility: int = 0,
        input_emb_dim: int = 128,
        output_emb_dim: int = 128,  # dimension of the output embeddings
        attention_layer_configs: Optional[dict] = None,
        external_pos: bool = False,
        cond_method: str = "add",
    ):
        # Skip the init of the AutoregressiveWrapper class
        super(AutoregressiveWrapper, self).__init__()

        if attention_layer_configs is None:
            attention_layer_configs = {}

        if not online:
            output_emb_dim = dim
        self.online = online

        if online and future_visibility > 0:
            max_seq_len += future_visibility

        self.output_emb = DelayPatternEmbedder(
            dim=dim,
            num_tokens=num_tokens,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
        )
        self.external_pos = external_pos

        if external_pos:
            self.input_pos_enc = ScaledSinusoidalEmbedding(input_emb_dim)
            self.output_pos_enc = ScaledSinusoidalEmbedding(output_emb_dim)

        print("\n")
        print(f"{'Parameter':<15} | {'Value'}")
        print("-" * 40)
        print(f"{'Dim':<15} | {dim}")
        print(f"{'Depth':<15} | {depth}")
        print(f"{'Heads':<15} | {heads}")
        print(f"{'Attn Dropout':<15} | {attn_dropout}")
        print(f"{'FF Dropout':<15} | {ff_dropout}")
        print(f"{'CrossAttend':<15} | {cross_attend}")
        print(f"{'Num Tokens':<15} | {num_tokens}")
        print(f"{'Num RVQ Layers':<15} | {num_rvq_layers}")
        print(f"{'Max Seq Len':<15} | {max_seq_len}")
        print(f"{'External Pos':<15} | {external_pos}")
        print(f"{'Future Visibility':<15} | {future_visibility}")
        print(f"{'Input Emb Dim':<15} | {input_emb_dim}")
        print(f"{'Shared':<15} | {shared}")
        print(f"{'Online':<15} | {online}")
        print(f"{'Cond Method':<15} | {cond_method}")

        if cond_method not in ("concat", "add", "film"):
            raise ValueError(
                f"Invalid conditioning method: {cond_method}. "
                f"Expected one of: concat, add, film"
            )

        self.decoder = TransformerWrapper(
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                cross_attend=cross_attend,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            token_emb=nn.Identity(),  # Needed to circumvent the 'embedding' layer.
            to_logits=MultiOutToLogits(num_rvq_layers, dim, num_tokens),
            use_abs_pos_emb=not external_pos,
        )

        # Use Delay Pattern
        delays = list(range(num_rvq_layers))
        self.pattern_provider = DelayedPatternProvider(
            n_q=num_rvq_layers, delays=delays
        )

        if online:
            self.future_visibility = future_visibility
            if self.future_visibility <= 0:
                # The padding is for when the model is generating
                #   without hearing the input
                self.input_padding = nn.Parameter(
                    torch.randn(-future_visibility + 1, input_emb_dim)
                )
                # Initialize the input_padding with kaiming normal
                nn.init.kaiming_normal_(self.input_padding)

            # Always store conditioning method and final LN flag
            self.cond_method = cond_method
            self.use_final_ln = True  # always use final LayerNorm

            # Only create dec_input_emb for concat conditioning
            if self.cond_method == "concat":
                self.dec_input_emb = nn.Linear(
                    input_emb_dim + output_emb_dim, dim
                )

            # For methods that need per-stream norms and projection ("add", "film")
            if self.cond_method in ("add", "film"):
                # LayerNorm on raw streams
                self.ln_in = nn.LayerNorm(input_emb_dim)
                self.ln_out = nn.LayerNorm(dim)

                # Project INPUT stream to OUTPUT/Transformer dim for fusion
                self.proj_in_to_out = (
                    nn.Linear(input_emb_dim, dim)
                    if input_emb_dim != dim
                    else nn.Identity()
                )

                # Optional final LayerNorm after fusion (over dim)
                if self.use_final_ln:
                    self.ln_fuse = nn.LayerNorm(dim)

            # For "add" path only: per-channel gate over dim
            if self.cond_method == "add":
                self.add_gate = nn.Parameter(torch.zeros(dim))

            # For "film" path only: generate (gamma, beta) in dim space
            elif self.cond_method == "film":
                self.film_gen = nn.Sequential(
                    nn.Linear(2 * dim, 4 * dim),
                    nn.GELU(),
                    nn.Linear(4 * dim, 2 * dim),
                )
                _last = self.film_gen[-1]
                if isinstance(_last, nn.Linear):
                    nn.init.zeros_(_last.weight)
                    nn.init.zeros_(_last.bias)

        # Save arguments
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len + 1  # +1 for the patterning token
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

        self.num_rvq_layers = num_rvq_layers

        # token used to temporarily pattern things integer id. Will be replaced by inst_tokens.
        self.temp_token = self.num_tokens
        self.shared = shared

    @property
    def net(self):
        return self.decoder

    def pad_input_embs_for_delay(self, input_emb):
        assert self.future_visibility <= 0
        input_pad = self.input_padding.expand(input_emb.shape[0], -1, -1)
        input_emb = torch.cat([input_pad, input_emb], dim=1)
        if self.future_visibility < 0:
            input_emb = input_emb[:, : self.future_visibility]
        return input_emb

    def pad_output_tokens_for_delay(self, output_tokens, trim_end=True):
        assert self.future_visibility > 0
        output_pad = torch.full(
            (
                output_tokens.shape[0],
                output_tokens.shape[1],
                self.future_visibility - 1,
            ),
            self.pad_value,
            device=output_tokens.device,
        )
        output_tokens = torch.cat([output_pad, output_tokens], dim=2)
        if trim_end:
            output_tokens = output_tokens[:, :, : -self.future_visibility]
        return output_tokens

    def get_input_embedding(self, output_tokens, input_emb):
        embedded_output = self.output_emb(output_tokens)

        # Optional external positional encodings (kept before normalization)
        if self.external_pos:
            input_pos, output_pos = generate_positions(
                length=input_emb.shape[1],
                delay=-self.future_visibility,
                device=embedded_output.device,
            )
            output_pos_emb = self.output_pos_enc(output_tokens, pos=output_pos)
            embedded_output = embedded_output + output_pos_emb
            input_pos_emb = self.input_pos_enc(input_emb, pos=input_pos)
            input_emb = input_emb + input_pos_emb

        # Backward compatibility: default to old concat behavior
        cond_method = getattr(self, "cond_method", "concat")

        if cond_method == "concat":
            concat_emb = torch.cat([input_emb, embedded_output], dim=-1)
            embedded = self.dec_input_emb(concat_emb)
            return embedded

        # "add" and "film" fuse in model dim
        if cond_method == "add":
            x_in = self.ln_in(input_emb)
            y_out = self.ln_out(embedded_output)
            x_proj = self.proj_in_to_out(x_in)
            x_fused = x_proj + y_out * self.add_gate
            if self.use_final_ln:
                x_fused = self.ln_fuse(x_fused)
            embedded = x_fused  # already [B, S, dim]
            return embedded

        if cond_method == "film":
            x_in = self.ln_in(input_emb)
            y_out = self.ln_out(embedded_output)
            x_proj = self.proj_in_to_out(x_in)
            h = torch.cat([x_proj, y_out], dim=-1)
            gamma_beta = self.film_gen(h)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            gamma = 1.0 + gamma
            x_fused = gamma * x_proj + beta
            if self.use_final_ln:
                x_fused = self.ln_fuse(x_fused)
            embedded = x_fused  # already [B, S, dim]
            return embedded

        # If an unknown method is specified, fall back to concat for safety
        concat_emb = torch.cat([input_emb, embedded_output], dim=-1)
        embedded = self.dec_input_emb(concat_emb)
        return embedded

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inst_tokens: Optional[torch.Tensor] = None,
        input_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder, used for training.

        Args:
            x (Tensor): Shape (B, K, T):

        Output:
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
        """
        B, K, T = x.shape
        pattern = self.pattern_provider.get_pattern(T)
        x, _, sequence_mask = pattern.build_pattern_sequence(
            x,
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )

        # Replace temporary patterning tokens with inst_tokens
        x = self._replace_temp_token_with_inst_tokens(x, inst_tokens)

        if self.online:
            if self.future_visibility <= 0:
                input_emb = self.pad_input_embs_for_delay(input_emb)
            else:
                x = self.pad_output_tokens_for_delay(x)

            embedded = self.get_input_embedding(x, input_emb)
        else:
            embedded = self.output_emb(x)

        logits = self.decoder(
            embedded, mask=mask, **kwargs
        )  # [B, K, S, num_tokens]

        logits = logits.permute(0, 3, 1, 2)  # [B, num_tokens, K, S]

        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, float("nan"), keep_only_valid_steps=True
        )

        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, num_tokens]

        logits_mask = logits_mask.unsqueeze(0).expand(B, *logits_mask.shape)
        return logits, logits_mask

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        seq_len: int,
        seq_out_start: Optional[torch.Tensor] = None,
        input_emb: None | torch.Tensor = None,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        display_pbar: bool = False,
        inst_tokens: torch.Tensor = None,
        guidance_scale: float = 1.0,  # parameter for classifier free guidance
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate function for multi-RVQ layer decoder models with delay pattern
        support.

        Args:
            seq_len (int): Number of steps to generate
            seq_out_start (torch.Tensor): Start index of the output sequence.
            input_emb (torch.Tensor): Input embeddings of the input context.
            temperature (float): Sampling temperature
            filter_logits_fn: Logit filtering function(s)
            filter_kwargs: Arguments for filtering functions
            cache_kv (bool): Cache key/value pairs
            display_pbar (bool): Show progress bar
            inst_tokens (torch.Tensor): instrument ids to use as prompt
            guidance_scale (float): Guidance scale for classifier free guidance.

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # Validate parameters and get batch_size and device
        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale
        )

        # Process filter functions
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )

        # Initialize pattern and output sequence
        out, prompt_length, post_mask, patterned_seq_out, seq_out_mask = (
            self._initialize_generation_pattern(
                seq_len, batch_size, device, seq_out_start, inst_tokens
            )
        )

        # Set flags
        greedy = temperature == 0.0

        # Initialize cache for KV caching
        cache = None

        # Handle online mode future visibility
        if self.online and self.future_visibility <= 0:
            input_emb = self.pad_input_embs_for_delay(input_emb)

        pbar = tqdm(
            range(prompt_length, seq_len),
            disable=not display_pbar,
            desc="Sampling",
        )
        for curr_sample_step in pbar:
            # Prepare model input (online-specific logic)
            if self.online:
                if self.future_visibility > 0:
                    # Here we add the left padding to the output tokens:
                    #   periods that model only need to listen but not generate.
                    # Later we will remove the padding before next step generation.
                    out = self.pad_output_tokens_for_delay(out, trim_end=False)
                if (
                    out.shape[-1] > input_emb.shape[1]
                ):  # consumed all the input embeddings
                    break
                cur_input_emb = input_emb[:, : out.shape[-1]]
                embedded = self.get_input_embedding(out, cur_input_emb)
            else:
                embedded = self.output_emb(out)  # [B, S, D]

            # Forward pass with optional KV caching
            if cache_kv:
                logits, intermediates = self.net(
                    embedded,
                    mask=None,
                    cache=cache,
                    return_intermediates=True,
                    **kwargs,
                )
                # Update cache from intermediates
                if hasattr(self.net, "can_cache_kv") and self.net.can_cache_kv:
                    cache = intermediates
            else:
                logits = self.net(
                    embedded,
                    mask=None,
                    return_intermediates=False,
                    **kwargs,
                )

            logits = logits[:, :, -1]  # Get logits for next token

            # Sample next tokens using mixin method
            samples = self._sample_next_tokens(
                logits,
                temperature,
                greedy,
                filter_logits_fn,
                filter_kwargs,
                curr_sample_step,
                out.shape[-1],
                guidance_scale,
            )

            out = torch.cat([out, samples], dim=-1)

            # Handle online mode future visibility
            if self.online and self.future_visibility > 0:
                out = out[
                    :, :, self.future_visibility - 1 :
                ]  # remove the padding for positive future visibility

            # Apply pattern constraints using mixin method
            out = self._apply_pattern_constraints(
                out,
                curr_sample_step,
                seq_out_start,
                patterned_seq_out,
                seq_out_mask,
                post_mask,
                inst_tokens,
            )

        # Finalize generation using mixin method
        return self._finalize_generation(out, guidance_scale)

    def _replace_temp_token_with_inst_tokens(
        self, x: torch.Tensor, inst_tokens: torch.Tensor
    ) -> torch.Tensor:
        temp_token_mask = x == self.temp_token
        inst_tokens_expanded = inst_tokens.view(x.shape[0], 1, 1).expand_as(x)
        x[temp_token_mask] = inst_tokens_expanded[temp_token_mask]
        return x


class EncoderDecoderTransformerMultiOut(nn.Module):
    """Encoder-decoder transformer model.

    This class is almost the same as x_transformers.XTransformer, but leaves
    flexibility to modify the network architecture and condition input method.

    Args:
        num_rvq_layers (int): Number of RVQ layers in decoder
        shared (bool): if the token ids in each RVQ level have the same range.

        duplicate_input_in_dec (bool): If True, the input stream is also
            duplicated in the decoder input, similar to the online case
        dec_output_emb_dim (int): Dimension to project the output stream to before
            concatenating with the input stream in the decoder.
            Only in use when duplicate_input_in_dec is True.
        Other args are the same as in models.py
    """

    def __init__(
        self,
        enc_dim: int = 512,
        dec_dim: int = 512,
        enc_depth: int = 6,
        dec_depth: int = 6,
        enc_heads: int = 8,
        dec_heads: int = 8,
        enc_num_tokens: int = 1024,
        dec_num_tokens: int = 1024,
        enc_max_seq_len: int = 512,
        dec_max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        input_emb_dim: int = 128,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,
        duplicate_input_in_dec: bool = False,
        dec_output_emb_dim: int = 128,
        attention_layer_configs: Optional[dict] = None,
    ):
        super().__init__()
        if attention_layer_configs is None:
            attention_layer_configs = {}

        self.input_emb_dim = input_emb_dim
        encoder_emb = nn.Linear(input_emb_dim, enc_dim)

        self.encoder = TransformerWrapperNoInitEmb(
            attn_layers=Encoder(
                dim=enc_dim,
                depth=enc_depth,
                heads=enc_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=enc_num_tokens,
            max_seq_len=enc_max_seq_len,
            return_only_embed=True,
            token_emb=encoder_emb,
        )
        self.decoder = DecoderTransformerMultiOut(
            dim=dec_dim,
            depth=dec_depth,
            heads=dec_heads,
            num_tokens=dec_num_tokens,
            max_seq_len=dec_max_seq_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pad_value=pad_value,
            cross_attend=True,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
            online=duplicate_input_in_dec,
            future_visibility=0,
            input_emb_dim=input_emb_dim,
            output_emb_dim=dec_dim,
            attention_layer_configs=attention_layer_configs,
        )

        # Save arguments
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_heads = enc_heads
        self.dec_heads = dec_heads
        self.enc_num_tokens = enc_num_tokens
        self.dec_num_tokens = dec_num_tokens
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

        self.num_rvq_layers = num_rvq_layers
        self.shared = shared

        self.duplicate_input_in_dec = duplicate_input_in_dec

    def forward(
        self,
        x_enc,
        x_dec,
        enc_mask=None,
        dec_mask=None,
        return_attn_z_loss=False,
        dec_inst_tokens=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder-decoder.

        Returns (See DecoderTransformerMultiOut):
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
            Optional attn_z_loss
        """
        if return_attn_z_loss:
            enc, cache = self.encoder(
                x_enc,
                mask=enc_mask,
                return_embeddings=True,
                return_attn_z_loss=True,
            )
            z_loss_enc = cache.attn_z_loss
            dec, cache = self.decoder(
                x_dec,
                context=enc,
                context_mask=enc_mask,
                mask=dec_mask,
                return_attn_z_loss=True,
            )
            z_loss_dec = cache.attn_z_loss
            return dec, z_loss_enc + z_loss_dec
        else:
            enc = self.encoder(x_enc, mask=enc_mask, return_embeddings=True)
            input_emb = x_enc if self.duplicate_input_in_dec else None
            dec = self.decoder(
                x_dec,
                context=enc,
                context_mask=enc_mask,
                mask=dec_mask,
                inst_tokens=dec_inst_tokens,
                input_emb=input_emb,
            )
            return dec

    @torch.no_grad()
    def generate(
        self,
        seq_in,
        seq_len,
        seq_out_start=None,
        mask=None,
        attn_mask=None,
        dec_inst_tokens=None,
        guidance_scale: float = 1.0,  # parameter for classifier free guidance
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            seq_in: Encoder input (B, T_enc)
            seq_len: Output sequence length
            mask: Encoder attention mask
            attn_mask: Encoder positional mask

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        encodings = self.encoder(
            seq_in, mask=mask, attn_mask=attn_mask, return_embeddings=True
        )
        input_emb = seq_in if self.duplicate_input_in_dec else None
        return self.decoder.generate(
            seq_len=seq_len,
            seq_out_start=seq_out_start,
            context=encodings,
            context_mask=mask,
            inst_tokens=dec_inst_tokens,
            input_emb=input_emb,
            guidance_scale=guidance_scale,
            **kwargs,
        )


class PrefixDecoderTransformerMultiOut(
    AutoregressiveWrapper, BaseGenerationMixin
):
    """
    Simple implementation of a prefix decoder.
    This is a offline decoder-only model that takes input as prefix and generates the output.

    Transform Decoder with support for multiple RVQ outputs and delay patterns
    Implements staggered generation strategy for parallel RVQ layer prediction.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers to predict
        shared (bool): if the token ids in each RVQ level have the same range.

        online (bool): If True, the model is in online mode and takes in concatenated
            input and output embeddings.
        future_visibility (int): Number of tokens to delay the input stream.
        input_emb_dim (int): Dimension of the input stream embeddings before concatenation.
        output_emb_dim (int): Dimension of the output stream embeddings before concatenation.

        Other args identical to models.py
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,  # Modified
        input_emb_dim: int = 128,
        attention_layer_configs: Optional[dict] = None,
    ):

        # Skip the init of the AutoregressiveWrapper class
        super(AutoregressiveWrapper, self).__init__()

        if attention_layer_configs is None:
            attention_layer_configs = {}

        self.output_emb = DelayPatternEmbedder(
            dim=dim,
            num_tokens=num_tokens,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
        )

        self.input_emb_dim = input_emb_dim
        self.input_emb_fc = nn.Linear(input_emb_dim, dim)

        print("\n")
        print(f"{'Parameter':<15} | {'Value'}")
        print("-" * 40)
        print(f"{'Dim':<15} | {dim}")
        print(f"{'Depth':<15} | {depth}")
        print(f"{'Heads':<15} | {heads}")
        print(f"{'Attn Dropout':<15} | {attn_dropout}")
        print(f"{'FF Dropout':<15} | {ff_dropout}")
        print(f"{'CrossAttend':<15} | {cross_attend}")
        print(f"{'Num Tokens':<15} | {num_tokens}")
        print(f"{'Num RVQ Layers':<15} | {num_rvq_layers}")
        print(f"{'Max Seq Len':<15} | {max_seq_len}")

        self.decoder = TransformerWrapper(
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                cross_attend=cross_attend,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            token_emb=nn.Identity(),  # Needed to circumvent the 'embedding' layer.
            to_logits=MultiOutToLogits(num_rvq_layers, dim, num_tokens),
        )

        # Use Delay Pattern
        delays = list(range(num_rvq_layers))
        self.pattern_provider = DelayedPatternProvider(
            n_q=num_rvq_layers, delays=delays
        )

        # Save arguments
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len + 1  # +1 for the pattern token
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

        self.num_rvq_layers = num_rvq_layers

        # token used to temporarily pattern things integer id. Will be replaced by inst_tokens.
        self.temp_token = self.num_tokens
        self.shared = shared

    @property
    def net(self):
        return self.decoder

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inst_tokens: Optional[torch.Tensor] = None,
        input_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder, used for training.

        Args:
            x (Tensor): Shape (B, K, T):

        Output:
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
        """

        B, K, T = x.shape
        pattern = self.pattern_provider.get_pattern(T)
        x, _, sequence_mask = pattern.build_pattern_sequence(
            x,
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )

        # Replace temporary patterning tokens with inst_tokens
        x = self._replace_temp_token_with_inst_tokens(x, inst_tokens)

        output_embedded = self.output_emb(x)
        input_embedded = self.input_emb_fc(input_emb)
        prefix_length = input_emb.shape[1]

        model_input = torch.cat([input_embedded, output_embedded], dim=1)

        logits = self.decoder(
            model_input, mask=mask, **kwargs
        )  # [B, K, S, num_tokens]

        logits = logits[:, :, prefix_length:, :]

        logits = logits.permute(0, 3, 1, 2)  # [B, num_tokens, K, S]

        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, float("nan"), keep_only_valid_steps=True
        )

        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, num_tokens]

        logits_mask = logits_mask.unsqueeze(0).expand(B, *logits_mask.shape)
        return logits, logits_mask

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        seq_len: int,
        seq_out_start: Optional[torch.Tensor] = None,
        input_emb: None | torch.Tensor = None,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        display_pbar: bool = False,
        inst_tokens: torch.Tensor = None,
        guidance_scale: float = 1.0,  # parameter for classifier free guidance
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate function for multi-RVQ layer decoder models with delay pattern
        support.

        Args:
            seq_len (int): Number of steps to generate
            batch_size (int): Number of sequences to generate
            temperature (float): Sampling temperature
            filter_logits_fn: Logit filtering function(s)
            filter_kwargs: Arguments for filtering functions
            cache_kv (bool): Cache key/value pairs
            display_pbar (bool): Show progress bar
            inst_tokens (torch.Tensor): instrument ids to use as prompt
            guidance_scale (float): Classifier-free guidance scale

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # Validate parameters and get batch_size and device
        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale
        )

        # Process filter functions
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )

        # Initialize pattern and output sequence
        out, prompt_length, post_mask, patterned_seq_out, seq_out_mask = (
            self._initialize_generation_pattern(
                seq_len, batch_size, device, seq_out_start, inst_tokens
            )
        )

        # Set flags
        greedy = temperature == 0.0

        # Initialize cache for KV caching
        cache = None

        # Precompute input embeddings for prefix decoder
        input_embedded = self.input_emb_fc(input_emb)

        pbar = tqdm(
            range(prompt_length, seq_len),
            disable=not display_pbar,
            desc="Sampling",
        )
        for curr_sample_step in pbar:
            # Prepare model input (prefix-specific logic)
            embedded = self.output_emb(out)  # [B, S, D]
            model_input = torch.cat([input_embedded, embedded], dim=1)

            # Forward pass with optional KV caching
            if cache_kv:
                logits, intermediates = self.net(
                    model_input,
                    mask=None,
                    cache=cache,
                    return_intermediates=True,
                    **kwargs,
                )
                # Update cache from intermediates
                if hasattr(self.net, "can_cache_kv") and self.net.can_cache_kv:
                    cache = intermediates
            else:
                logits = self.net(
                    model_input,
                    mask=None,
                    return_intermediates=False,
                    **kwargs,
                )

            logits = logits[:, :, -1]  # Get logits for next token

            # Sample next tokens using mixin method
            samples = self._sample_next_tokens(
                logits,
                temperature,
                greedy,
                filter_logits_fn,
                filter_kwargs,
                curr_sample_step,
                out.shape[-1],
                guidance_scale,
            )

            out = torch.cat([out, samples], dim=-1)

            # Apply pattern constraints using mixin method
            out = self._apply_pattern_constraints(
                out,
                curr_sample_step,
                seq_out_start,
                patterned_seq_out,
                seq_out_mask,
                post_mask,
                inst_tokens,
            )

        # Finalize generation using mixin method
        return self._finalize_generation(out, guidance_scale)

    def _replace_temp_token_with_inst_tokens(
        self, x: torch.Tensor, inst_tokens: torch.Tensor
    ) -> torch.Tensor:
        temp_token_mask = x == self.temp_token
        inst_tokens_expanded = inst_tokens.view(x.shape[0], 1, 1).expand_as(x)
        x[temp_token_mask] = inst_tokens_expanded[temp_token_mask]
        return x


class OnlinePrefixDecoderTransformerMultiOut(DecoderTransformerMultiOut):
    """
    Simple implementation of an online chunked prediction model by prefix decoder.
    This is a online decoder-only model that takes input context,
    and its own previous output as prefix and generates a chunk of prediction.

    Transform Decoder with support for multiple RVQ outputs and delay patterns
    Implements staggered generation strategy for parallel RVQ layer prediction.

    Attributes:
        num_rvq_layers (int): Number of RVQ layers to predict
        shared (bool): if the token ids in each RVQ level have the same range.

        online (bool): If True, the model is in online mode and takes in concatenated
            input and output embeddings.
        future_visibility (int): Number of tokens to delay the input stream.
        input_emb_dim (int): Dimension of the input stream embeddings before concatenation.
        output_emb_dim (int): Dimension of the output stream embeddings before concatenation.

        Other args identical to models.py
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        num_rvq_layers: int = 4,  # Modified
        shared: bool = True,  # Modified
        input_emb_dim: int = 128,
        attention_layer_configs: Optional[dict] = None,
        future_visibility: int = 0,
        output_emb_dim: int = 128,
        chunk_length: int = 10,
        chunk_start_prob: float = 0.1,
    ):
        # init with DecoderTransformerMultiOut's init
        # The max_seq_len will be larger than the actual sequence length seen by the model,
        # but that's fine.
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pad_value=pad_value,
            cross_attend=cross_attend,
            num_rvq_layers=num_rvq_layers,
            shared=shared,
            online=True,  # Always online for this class
            future_visibility=future_visibility,
            input_emb_dim=input_emb_dim,
            output_emb_dim=output_emb_dim,
            attention_layer_configs=attention_layer_configs,
            external_pos=False,  # Always disable external position encoding for now.
            cond_method="add",  # Always use add method for now.
        )

        self.chunk_length = chunk_length
        self.chunk_start_prob = chunk_start_prob

    @property
    def net(self):
        return self.decoder

    def prepare_input(
        self,
        x: torch.Tensor,
        inst_tokens: torch.Tensor,
        input_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int, torch.Tensor, torch.Tensor]:
        """Prepare input embeddings and output tokens for online training.

        This method handles chunking the input sequence into smaller segments for online
        training, applies delay patterns, and prepares the input embeddings and output
        tokens for the forward pass.

        Args:
            x (torch.Tensor): Output tokens of shape [B, K, T] where B is batch size,
                K is number of RVQ layers, and T is sequence length.
            inst_tokens (torch.Tensor): Instrument tokens for each batch item.
            input_emb (torch.Tensor): Input embeddings of shape [B, T, D] where D is
                the embedding dimension.

        Returns:
            Tuple containing prepared inputs for the model forward pass.
        """

        # Decide output context length (which inner-loop we are in)
        B, K, T = x.shape
        # Apply delay pattern to the output
        pattern = self.pattern_provider.get_pattern(T)
        x_patterned, _, sequence_mask = pattern.build_pattern_sequence(
            x,
            special_token=self.temp_token,
            keep_only_valid_steps=True,
        )
        # Replace temporary patterning tokens with inst_tokens
        x_patterned = self._replace_temp_token_with_inst_tokens(
            x_patterned, inst_tokens
        )

        # Here we assume the max duration of generation
        # is divisible by chunk_length.
        if T % self.chunk_length != 0:
            raise ValueError(
                f"Sequence length T={T} must be divisible by chunk_length={self.chunk_length}."
            )

        max_duration_gen = T - max(0, self.future_visibility)
        possible_context_lengths = np.arange(
            0, max_duration_gen, self.chunk_length
        )
        context_length = np.random.choice(possible_context_lengths)

        # The start index is always 0.
        # Because the input is randomly chunked, we do not random the start index.
        context_end_idx = context_length

        # Adjust input and output according to future visibility, and get input and output context
        if self.future_visibility <= 0:
            input_emb = self.pad_input_embs_for_delay(input_emb)
            # Here we +1 to include the BOS.
            context_end_idx += 1
        else:
            x_patterned = self.pad_output_tokens_for_delay(x_patterned)
            # For future_visibility > 0,
            # BOS is included in self.future_visibility tokens in the beginning.
            context_end_idx += self.future_visibility

        input_context = input_emb[:, :context_end_idx, :]
        output_context = x_patterned[:, :, :context_end_idx]
        output = x_patterned[
            :, :, context_end_idx : context_end_idx + self.chunk_length
        ]
        targets = x_patterned[
            :, :, context_end_idx : context_end_idx + self.chunk_length
        ]
        pred_start_idx = context_end_idx - 1
        pred_end_idx = pred_start_idx + self.chunk_length

        assert input_context.shape[1] == output_context.shape[-1]

        in_embedded = self.get_input_embedding(output_context, input_context)
        out_embedded = self.ln_out(self.output_emb(output))
        embedded = torch.cat([in_embedded, out_embedded], dim=1)

        # sequence_mask is a triangle for the first 1 to N+1 steps
        #   because of delay pattern (e.g. 1st step only predict 1st rvq layer).
        # Here we +1 to skip the bos which is False accorss all rvq layers
        logits_mask = sequence_mask[:, pred_start_idx + 1 : pred_end_idx + 1]
        # Add batch dim
        logits_mask = logits_mask.unsqueeze(0).expand(
            input_emb.shape[0], *logits_mask.shape
        )

        assert logits_mask.shape[-1] == self.chunk_length
        assert targets.shape[-1] == self.chunk_length

        return (
            embedded,
            pred_start_idx,
            pred_end_idx,
            targets,
            logits_mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        inst_tokens: torch.Tensor,
        input_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder, used for training.

        Args:
            x (Tensor): Shape (B, K, T):

        Output:
            logits (Tensor): output logits, shape (B, K, T, num_tokens).
            logits_mask: output logits mask, such that logits[logits_mask]
                provides only valid logits. Shape (B, K, T)
        """
        (
            embedded,
            pred_start_idx,
            pred_end_idx,
            targets,
            logits_mask,
        ) = self.prepare_input(x, inst_tokens, input_emb)

        logits = self.decoder(
            embedded, mask=mask, **kwargs
        )  # [B, K, S, num_tokens]

        logits_pred = logits[:, :, pred_start_idx:pred_end_idx, :]
        assert logits_pred.shape[2] > 0
        assert logits_mask.shape[2] == logits_pred.shape[2]
        assert logits_pred.shape[2] == targets.shape[2]
        return logits_pred, logits_mask, targets

    def _apply_pattern_constraints(
        self,
        out: torch.Tensor,
        curr_sample_step: int,
        curr_global_step: int,
        post_mask: torch.Tensor,
        inst_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply pattern-based constraints for online prefix decoder generation.

        The indexing is a bit different than that in decoder online model and prefix decoder.
        """
        # +1 because we start with 1 token (bos / instrument token)
        replace_mask = ~post_mask[..., curr_global_step + 1]
        if replace_mask.any():
            out[:, replace_mask, curr_sample_step] = self.temp_token
            # Replace pattern token with instrument id
            out = self._replace_temp_token_with_inst_tokens(out, inst_tokens)
        return out

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate_chunk(
        self,
        global_start_idx: int,
        context_emb: torch.Tensor,
        generate_len: int,
        inst_tokens: torch.Tensor,
        post_mask: torch.Tensor,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inner-loop generate function for multi-RVQ layer decoder models with delay pattern
        support. This function generates a chunk of output tokens given the input context.
        The input context contains the input tokens and the output tokens before the chunk.
        The output context contains the output tokens after the chunk.

        Args:
            global_start_idx (int): Global start index of the chunk.
            context_emb (torch.Tensor): Embeddings of the input context,
                contains both input and previous output.
            generate_len (int): Number of steps to generate.
            inst_tokens (torch.Tensor): Instrument tokens for each batch item.
            post_mask (torch.Tensor): Mask for the post-patterning tokens.
                Mainly for indicating where are the instrument tokens (act as BOS) at the
                beginning, and thus need to fill in instrument tokens instead of using
                the the sampled output token.
            temperature (float): Sampling temperature.
            filter_logits_fn: Logit filtering function(s).
            filter_kwargs: Arguments for filtering functions.
            cache_kv (bool): Cache key/value pairs.
            **kwargs: Additional arguments for the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Generated output tokens and model input.
        """
        # Set flags
        greedy = temperature == 0.0

        # Initialize cache for KV caching
        cache = None

        # No need to precompute input embeddings as it is prepared by the outer loop.
        out = torch.zeros(
            context_emb.shape[0],
            self.num_rvq_layers,
            0,
            device=context_emb.device,
            dtype=torch.long,
        )

        model_input = context_emb

        for curr_sample_step in range(generate_len):
            # Forward pass with optional KV caching
            if cache_kv:
                logits, intermediates = self.net(
                    model_input,
                    mask=None,
                    cache=cache,
                    return_intermediates=True,
                    **kwargs,
                )
                # Update cache from intermediates
                if hasattr(self.net, "can_cache_kv") and self.net.can_cache_kv:
                    cache = intermediates
            else:
                logits = self.net(
                    model_input,
                    mask=None,
                    return_intermediates=False,
                    **kwargs,
                )

            logits = logits[:, :, -1]  # Get logits for next token

            # Sample next tokens using mixin method
            samples = self._sample_next_tokens(
                logits,
                temperature,
                greedy,
                filter_logits_fn,
                filter_kwargs,
                curr_sample_step,
                out.shape[-1],
                guidance_scale=1.0,  # disable guidance for now
            )

            # append the samples to the output
            out = torch.cat([out, samples], dim=-1)

            # Apply pattern constraints using mixin method
            out = self._apply_pattern_constraints(
                out,
                curr_sample_step,
                curr_global_step=global_start_idx + curr_sample_step,
                post_mask=post_mask,
                inst_tokens=inst_tokens,
            )

            # append the samples to the model input
            out_embedded = self.ln_out(self.output_emb(out))
            model_input = torch.cat([model_input, out_embedded], dim=1)

        return out

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        seq_len: int,
        seq_out_start: Optional[torch.Tensor] = None,
        input_emb: None | torch.Tensor = None,
        temperature: float = 1.0,
        filter_logits_fn: (
            str | Callable | list[str | Callable]
        ) = top_k_multi_out,
        filter_kwargs: dict | list[dict] = dict(),
        cache_kv: bool = True,
        display_pbar: bool = False,
        inst_tokens: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate function for Online Prefix Decoder that
            predicts a chunk of multi-RVQ audio tokens in delay pattern.

        Args:
            seq_len (int): Number of steps to generate.
            seq_out_start (torch.Tensor): Start index of the output sequence.
            input_emb (torch.Tensor): Input embeddings of the input context.
            temperature (float): Sampling temperature
            filter_logits_fn: Logit filtering function(s)
            filter_kwargs: Arguments for filtering functions
            cache_kv (bool): Cache key/value pairs
            display_pbar (bool): Show progress bar
            inst_tokens (torch.Tensor): instrument ids to use as prompt

        Returns:
            Tensor: Generated sequences (B, K, seq_len - 1)
        """
        # NOTE: we do not support prompting for now.
        if seq_out_start is not None:
            raise NotImplementedError("Prompting is not supported yet.")

        # Validate parameters and get batch_size and device.
        # seq_out_start: we do not support CFG for now.
        batch_size, device = self._validate_generation_params(
            inst_tokens, seq_out_start, input_emb, guidance_scale=1.0
        )

        # Process filter functions
        filter_logits_fn, filter_kwargs = self._process_filter_functions(
            filter_logits_fn, filter_kwargs
        )

        # Initialize pattern and output sequence
        (
            output_tokens,
            prompt_length,
            post_mask,
            patterned_seq_out,
            seq_out_mask,
        ) = self._initialize_generation_pattern(
            seq_len, batch_size, device, seq_out_start, inst_tokens
        )

        # Handle online mode future visibility
        if self.future_visibility <= 0:
            input_emb = self.pad_input_embs_for_delay(input_emb)
        else:
            output_tokens = self.pad_output_tokens_for_delay(
                output_tokens, trim_end=False
            )

        start_indices = np.arange(0, seq_len, self.chunk_length)
        gen_lengths = np.minimum(self.chunk_length, seq_len - start_indices)

        display_pbar = True

        pbar = tqdm(
            range(len(start_indices)),
            disable=not display_pbar,
            desc="Generating chunks",
        )

        for i in pbar:
            start_idx = start_indices[i]
            gen_length = gen_lengths[i]
            input_context = self.get_input_embedding(
                output_tokens,
                input_emb[:, : output_tokens.shape[-1], :],
            )

            chunk_output_tokens = self.generate_chunk(
                start_idx,
                input_context,
                gen_length,
                inst_tokens,
                post_mask,
                temperature,
                filter_logits_fn,
                filter_kwargs,
                cache_kv,
                **kwargs,
            )
            output_tokens = torch.cat(
                [output_tokens, chunk_output_tokens], dim=-1
            )

        # Handle positive future visibility
        if self.future_visibility > 0:
            delay_amount = self.future_visibility
            output_tokens = output_tokens[
                :, :, delay_amount:
            ]  # remove the padding for positive future visibility

        return self._finalize_generation(output_tokens)
