from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .convert import iter_quant_linears


@dataclass
class SensitivityItem:
    name: str
    score: float


def rank_linear_sensitivity(
    model: nn.Module,
    loss_fn: Callable[[nn.Module], torch.Tensor],
    limit: int | None = None,
) -> list[SensitivityItem]:
    """Rank QuantLinear modules by single-module quantization sensitivity.

    `loss_fn(model)` should return a scalar proxy loss, e.g. MSE between BF16
    and quantized DiT noise predictions over cached calibration batches.
    """

    modules = list(iter_quant_linears(model))
    if limit is not None:
        modules = modules[:limit]

    scores: list[SensitivityItem] = []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for name, module in modules:
            old = module.state.enabled
            for _, other in iter_quant_linears(model):
                other.state.enabled = False
            module.state.enabled = True
            score = float(loss_fn(model).detach().float().cpu().item())
            scores.append(SensitivityItem(name=name, score=score))
            module.state.enabled = old

        for _, module in iter_quant_linears(model):
            module.state.enabled = True

    if was_training:
        model.train()
    return sorted(scores, key=lambda item: item.score, reverse=True)
