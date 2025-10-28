"""Sampling functions."""

import torch
import torch.nn.functional as F
from math import ceil

from typing import Callable, List, Optional

# The following functions are modified from x_transformers.
# We just added the kwargs to the functions to make them compatible with
#   other inference-time inputs.


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# nucleus


def top_p(logits, thres=0.9, **kwargs):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(
        sorted_indices_to_remove, (1, -1), value=False
    )

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk


def top_k(logits, frac_num_tokens=0.1, k=None, **kwargs):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# For delay-pattern generation
def top_k_multi_out(logits, frac_num_tokens=0.1, k=None, **kwargs):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs

# top_a


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02, **kwargs):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float("-inf"), logits)


# min_p
# https://arxiv.org/abs/2407.01082


def min_p(logits, min_p=0.1, **kwargs):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float("-inf"), logits)


def rvq_mask_logits(
    logits,
    num_non_rvq_tokens: int = -1,
    num_codebook_per_layer: int = -1,
    num_rvq_layers: int = -1,
    curr_sample_step: int = 0,
    **kwargs,
):
    """Masked sampling for sampling flattened RVQ logits."""
    # logits: [B, C]
    if num_non_rvq_tokens == -1:
        raise ValueError("num_non_rvq_tokens must be provided.")
    if num_codebook_per_layer == -1:
        raise ValueError("num_codebook_per_layer must be provided.")
    if num_rvq_layers == -1:
        raise ValueError("num_rvq_layers must be provided.")
    num_tokens = num_codebook_per_layer * num_rvq_layers + num_non_rvq_tokens
    if num_tokens != logits.shape[-1]:
        raise ValueError(
            f"num_tokens {num_tokens} != logits.shape[-1] {logits.shape[-1]}"
        )

    # mask out the tokens that are not in the current layer
    curr_rvq_layer = curr_sample_step % num_rvq_layers
    mask = torch.zeros_like(logits)
    start_idx = curr_rvq_layer * num_codebook_per_layer + num_non_rvq_tokens
    end_idx = (curr_rvq_layer + 1) * num_codebook_per_layer + num_non_rvq_tokens
    mask[:, start_idx:end_idx] = 1
    logits.masked_fill_(mask == 0, float("-inf"))
    return logits


def validate_filter_fn_kwargs(
    filter_logits_fn: str | List[str],
    filter_kwargs: dict | List[dict],
):
    lists = [
        isinstance(filter_logits_fn, list),
        isinstance(filter_kwargs, list),
    ]
    if any(lists) and not all(lists):
        raise ValueError(
            "filter_logits_fn and filter_kwargs must both be lists or "
            "both be single items."
        )
    if all(lists) and len(filter_logits_fn) != len(filter_kwargs):
        raise ValueError(
            "filter_logits_fn and filter_kwargs must have the same length."
        )
    is_list = all(lists)
    return is_list


# filter logits functions dict[str -> Callable]

FILTER_LOGITS_FN = dict(
    top_p=top_p,
    top_k=top_k,
    top_k_multi_out=top_k_multi_out, # Modified
    top_a=top_a,
    min_p=min_p,
    rvq_mask_logits=rvq_mask_logits,
)


class ComposeFilterFns:
    def __init__(
        self,
        filter_fns: List[str | Callable],
        filter_kwargs: Optional[List[dict]] = None,
    ):
        if isinstance(filter_fns[0], str):
            filter_fns = [FILTER_LOGITS_FN[fn] for fn in filter_fns]
        self.filter_fns = filter_fns

        self.filter_kwargs = default(filter_kwargs, [{}] * len(filter_fns))

    def __call__(self, logits, **kwargs):
        for filter_fn, filter_kwargs in zip(
            self.filter_fns, self.filter_kwargs
        ):
            fn_kwargs = {**filter_kwargs, **kwargs}
            logits = filter_fn(logits, **fn_kwargs)
        return logits
