"""Utilities for load model and inference with a trained model."""

import torch
from typing import Optional
from pathlib import Path
import argbind

from stream_music_gen.base_trainer import BaseLightningModel


def load_lit_model(
    model_path: str,
    lit_module_cls: BaseLightningModel,
    batch_size: Optional[int] = None,
    override_args: Optional[dict] = None,
    compile: bool = True,
    return_only_model: bool = False,
    return_lit_module: bool = False,
):
    """Load model and dataloader via lightning module from a given model path."""
    model_path = Path(model_path)
    experiment_dir = model_path.parent

    print(f"Loading model from {model_path}")

    args = argbind.load_args(experiment_dir / "args.yml")
    if batch_size:
        # If batch_size is provided, override the one in the args
        args["batch_size"] = batch_size

    if not compile:
        args["compile"] = False

    if override_args:
        args.update(override_args)

    with argbind.scope(args):
        lit_module = lit_module_cls()
        train_dataloader, val_dataloader = lit_module.get_dataloaders()

    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )["state_dict"]

    if not args["compile"]:
        state_dict = {
            k.replace("._orig_mod", ""): v for k, v in state_dict.items()
        }

    lit_module.load_state_dict(state_dict)
    model = lit_module.model
    model.config = args  # Add config to model for saving

    tokenizer = lit_module.tokenizer
    dataloaders = train_dataloader, val_dataloader
    if return_only_model and return_lit_module:
        raise ValueError(
            "Only one of return_only_model and return_lit_module can be True."
        )
    if return_only_model:
        return model
    if return_lit_module:
        return lit_module

    return model, tokenizer, dataloaders
