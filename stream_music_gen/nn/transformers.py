"""Template class for Transformer models with all the tricks."""

import x_transformers
import copy
from typing import Optional

DEFAULT_KWARGS = {
    # pre-normalization: x-transformers has default pre-normalization
    # ff_multi: x-transformers has default ff dim 4x of hidden dim
    "rotary_pos_emb": True,  # https://arxiv.org/abs/2104.09864
    "use_simple_rmsnorm": True,  # https://arxiv.org/abs/2307.14995
    "attn_qk_norm": True,  # https://arxiv.org/abs/2010.04245
    "attn_qk_norm_dim_scale": True,  # https://arxiv.org/abs/2302.05442
    # swiglu: https://arxiv.org/abs/2002.05202
    "ff_swish": True,
    # "ff_glu": True,  # LLAMA 3.2 only use silu
    "attn_kv_heads": 8,  # https://arxiv.org/abs/2305.13245, used in llama 3
    # Flash attention: https://arxiv.org/abs/2205.14135
    # Need to build flash attention to enable it.
    # https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    "attn_flash": True,
    # T5 relative position bias cannot be used with flash attention
    # "rel_pos_bias": True,  # T5 relative position bias
}


def Decoder(
    dim: int,
    depth: int,
    heads: int,
    attn_dropout: float = 0.0,
    ff_dropout: float = 0.1,
    cross_attend: bool = True,
    attention_layer_configs: Optional[dict] = None,
):
    if attention_layer_configs is None:
        attention_layer_configs = {}
    attention_layer_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    attention_layer_kwargs.update(attention_layer_configs)
    return x_transformers.Decoder(
        dim=dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        cross_attend=cross_attend,
        **attention_layer_kwargs,
    )


def Encoder(
    dim: int,
    depth: int,
    heads: int,
    attn_dropout: float = 0.0,
    ff_dropout: float = 0.1,
    attention_layer_configs: Optional[dict] = None,
):
    if attention_layer_configs is None:
        attention_layer_configs = {}
    attention_layer_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    attention_layer_kwargs.update(attention_layer_configs)
    return x_transformers.Encoder(
        dim=dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        **attention_layer_kwargs,
    )
