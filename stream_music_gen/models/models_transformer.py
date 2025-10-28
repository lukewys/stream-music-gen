"""Models for online stem gen."""

from typing import Tuple, Callable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from x_transformers import TransformerWrapper, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    exists,
    join,
    align_right,
)
from tqdm import tqdm

from stream_music_gen.nn.transformers import Decoder, Encoder
from stream_music_gen.models.sampling import (
    top_k,
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    validate_filter_fn_kwargs,
)


class SumRVQEmb(nn.Module):
    """Sum the embeddings of RVQ tokens as embedding layers."""

    def __init__(self, dim: int, num_tokens: int, num_rvq_layers: int):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.num_rvq_layers = num_rvq_layers
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        # x: [b, t]
        x = x.reshape(x.shape[0], -1, self.num_rvq_layers)
        x = self.emb(x)
        x = x.sum(dim=-2)
        return x


class TransformerWrapperNoInitEmb(TransformerWrapper):
    """A transformer wrapper without initializing embedding layer."""

    def init_(self):
        pass


class DecoderTransformer(AutoregressiveWrapper):
    """Decoder-only transformer model."""

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
        attention_layer_configs: Optional[dict] = None,
        transformer_configs: Optional[dict] = None,
    ):
        # Skip the init of the AutoregressiveWrapper class
        super(AutoregressiveWrapper, self).__init__()

        if attention_layer_configs is None:
            attention_layer_configs = {}
        if transformer_configs is None:
            transformer_configs = {}

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
            **transformer_configs,
        )

        # Save arguments
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.pad_value = pad_value

    @property
    def net(self):
        return self.decoder

    def forward(self, x, mask=None, **kwargs):
        return self.decoder(x, mask=mask, **kwargs)

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        prompts,
        seq_len,
        eos_token=None,
        temperature=1.0,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable | List[str | Callable] = top_k,
        restrict_to_max_seq_len=True,
        filter_kwargs: dict | List[dict] = dict(),
        cache_kv=True,
        guidance_scale: float = 1.0,  # new parameter for classifier free guidance
        display_pbar=False,
        **kwargs,
    ):
        """Generate Function modified from AutoregressiveWrapper.

        Add the following supports:
        1. Input current sampling step to filter_logits_fn.
        2. support masked sampling.
        3. support classifier free guidance.
        4. Support tqdm progress bar.

        Remove the following supports:
        1. contrastive decoding

        Classifier-free guidance:

        If `guidance_scale != 1.0`, then it is assumed that the provided
        conditioning (passed via kwargs, e.g. as `context` and `context_mask`)
        has a doubled batch dimension: the first half is the real context,
        and the second half is the null (all-zeros) context.

        During generation the logits from both halves are combined as:
            logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
        and the resulting sample is duplicated so that the KV cache remains aligned.
        """
        # handle multiple filter logits functions
        filter_fns_is_list = validate_filter_fn_kwargs(
            filter_logits_fn, filter_kwargs
        )
        if filter_fns_is_list:
            filter_fns = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_logits_fn = filter_fns
            filter_kwargs = dict()

        max_seq_len, greedy, device = (
            self.max_seq_len,
            temperature == 0.0,
            prompts.device,
        )

        b, t = prompts.shape

        # if using CFG, we expect the batch to be doubled (first half cond, second half null)
        if guidance_scale != 1.0:
            assert b % 2 == 0, (
                "For classifier-free guidance, prompts (and corresponding context) "
                "must have an even batch size (first half conditioned, second half null)."
            )

        # handle filter logits fn given as string
        if isinstance(filter_logits_fn, str):
            assert (
                filter_logits_fn in FILTER_LOGITS_FN
            ), f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)
        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens are appended
        out = prompts

        # kv cache
        cache = None

        pbar = tqdm(
            range(seq_len),
            disable=not display_pbar,
            desc="Sampling",
        )

        for curr_sample_step in pbar:
            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len
                assert not (
                    cache_kv
                    and max_len_exceeded
                    and not self.net.can_cache_kv_outside_max_seq_len
                ), (
                    "the network cannot use cached key values when decoding outside "
                    "the max sequence length. consider switching to rotary embeddings."
                )
                x = out[:, -max_seq_len:]
                if exists(cache):
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == "a":
                            inter.cached_kv = [
                                t[..., -(max_seq_len - 1) :, :]
                                for t in inter.cached_kv
                            ]
            else:
                x = out

            logits, new_cache = self.net(
                x,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos,
                **kwargs,
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]  # get logits for the last token

            if guidance_scale != 1.0:
                # Expecting doubled batch: split into conditioned and null (unconditional)
                batch_size = logits.shape[0]
                half = batch_size // 2
                logits_cond = logits[:half]
                logits_uncond = logits[half:]
                # Combine the logits using the CFG formula:
                logits = logits_uncond + guidance_scale * (
                    logits_cond - logits_uncond
                )

            # Sampling step
            if greedy:
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                curr_sample_length = out.shape[-1]
                filtered_logits = filter_logits_fn(
                    logits,
                    curr_sample_step=curr_sample_step,
                    curr_sample_length=curr_sample_length,
                    **filter_kwargs,
                )
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            if guidance_scale != 1.0:
                # Duplicate the sampled token for both halves so that future KV cache and token sequences remain aligned.
                sample = torch.cat([sample, sample], dim=0)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(dim=-1).all():
                    break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        # Remove the prompt tokens
        out = out[:, t:]

        # If using CFG, return only the conditioned half.
        if guidance_scale != 1.0:
            half = out.shape[0] // 2
            out = out[:half]

        return out


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer model.

    This class is almost the same as x_transformers.XTransformer, but leaves
    flexibility to modify the network architecture and condition input method.
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
        attention_layer_configs: dict = None,
        transformer_configs: dict = None,
    ):
        super().__init__()
        if attention_layer_configs is None:
            attention_layer_configs = {}
        if transformer_configs is None:
            transformer_configs = {}

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
            **transformer_configs,
        )
        self.decoder = DecoderTransformer(
            dim=dec_dim,
            depth=dec_depth,
            heads=dec_heads,
            num_tokens=dec_num_tokens,
            max_seq_len=dec_max_seq_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pad_value=pad_value,
            cross_attend=True,
            attention_layer_configs=attention_layer_configs,
            transformer_configs=transformer_configs,
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

    def forward(
        self,
        x_enc,
        x_dec,
        enc_mask=None,
        dec_mask=None,
        return_attn_z_loss=False,
    ):
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
            dec = self.decoder(
                x_dec, context=enc, context_mask=enc_mask, mask=dec_mask
            )
            return dec

    @torch.no_grad()
    def generate(
        self,
        seq_in,
        seq_out_start,
        seq_len,
        mask=None,
        attn_mask=None,
        **kwargs,
    ):
        encodings = self.encoder(
            seq_in, mask=mask, attn_mask=attn_mask, return_embeddings=True
        )
        return self.decoder.generate(
            seq_out_start,
            seq_len,
            context=encodings,
            context_mask=mask,
            **kwargs,
        )
