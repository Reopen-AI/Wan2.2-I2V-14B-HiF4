from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.distributed as dist
from torch import nn

from .convert import iter_quant_linears
from .fp4_formats import quantile_clip_ratio
from .quant_linear import QuantLinear


@dataclass
class ActStat:
    amax: torch.Tensor
    mean_abs: torch.Tensor
    samples: int = 0


def merge_distributed_act_stats(stats: dict[str, ActStat]) -> dict[str, ActStat]:
    """Merge activation statistics across data-parallel ranks.

    `amax` is reduced by max. `mean_abs` is reduced as a weighted mean using the
    number of flattened activation rows observed by each rank.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return stats

    merged: dict[str, ActStat] = {}
    all_names = sorted(stats)
    gathered_names: list[list[str] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_names, all_names)
    names = sorted({name for rank_names in gathered_names for name in (rank_names or [])})

    for name in names:
        item = stats.get(name)
        local_shape = tuple(item.amax.shape) if item is not None else None
        shapes = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(shapes, local_shape)
        non_null_shapes = [candidate for candidate in shapes if candidate is not None]
        if not non_null_shapes:
            continue
        shape = non_null_shapes[0]
        if any(candidate != shape for candidate in non_null_shapes):
            raise ValueError(f"Distributed stat shape mismatch for {name}: {shapes}")

        if item is None:
            amax = torch.zeros(shape, dtype=torch.float32)
            mean_abs = torch.zeros(shape, dtype=torch.float32)
            samples = 0
        else:
            amax = item.amax.float().cpu()
            mean_abs = item.mean_abs.float().cpu()
            samples = int(item.samples)

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        amax_reduce = amax.to(device)
        dist.all_reduce(amax_reduce, op=dist.ReduceOp.MAX)

        weighted = (mean_abs * samples).to(device)
        count = torch.tensor([samples], dtype=torch.float64, device=device)
        dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total = int(count.item())
        mean_reduce = weighted / max(total, 1)

        merged[name] = ActStat(
            amax=amax_reduce.cpu(),
            mean_abs=mean_reduce.cpu(),
            samples=total,
        )

    return merged


def collect_linear_input_stats(
    model: nn.Module,
    run_calibration: Callable[[nn.Module], None],
    max_modules: int | None = None,
    disable_quant: bool = True,
) -> dict[str, ActStat]:
    """Collect per-input-channel stats for all QuantLinear modules.

    `run_calibration(model)` should execute representative Wan2.2 denoising
    calls with `torch.no_grad()`. This keeps the code independent from the
    official repository's pipeline class.
    """

    stats: dict[str, ActStat] = {}
    handles = []

    def make_hook(name: str):
        def hook(module: QuantLinear, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            amax = flat.abs().amax(dim=0).cpu()
            mean_abs = flat.abs().mean(dim=0).cpu()
            if name not in stats:
                stats[name] = ActStat(amax=amax, mean_abs=mean_abs, samples=flat.shape[0])
            else:
                item = stats[name]
                item.amax = torch.maximum(item.amax, amax)
                item.mean_abs = (
                    item.mean_abs * item.samples + mean_abs * flat.shape[0]
                ) / (item.samples + flat.shape[0])
                item.samples += flat.shape[0]

        return hook

    for idx, (name, module) in enumerate(iter_quant_linears(model)):
        if max_modules is not None and idx >= max_modules:
            break
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    saved_enabled: dict[str, bool] = {}
    if disable_quant:
        for name, module in iter_quant_linears(model):
            saved_enabled[name] = module.state.enabled
            module.state.enabled = False

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            run_calibration(model)
    finally:
        if was_training:
            model.train()
        for handle in handles:
            handle.remove()
        for name, module in iter_quant_linears(model):
            if name in saved_enabled:
                module.state.enabled = saved_enabled[name]
    return stats


def apply_smoothquant(
    model: nn.Module,
    stats: dict[str, ActStat],
    alpha: float = 0.5,
    min_scale: float = 1e-4,
    max_scale: float = 1e4,
) -> None:
    """Apply SmoothQuant-style equivalent scaling to QuantLinear modules."""

    for name, module in iter_quant_linears(model):
        if name not in stats:
            continue
        act_amax = stats[name].amax.to(module.weight.device).float().clamp_min(min_scale)
        weight_amax = module.weight.detach().float().abs().amax(dim=0).clamp_min(min_scale)
        scale = (act_amax.pow(alpha) / weight_amax.pow(1.0 - alpha)).clamp(min_scale, max_scale)
        module.state.smooth_scale = scale.cpu()
        module.invalidate_cache()


def tune_clip_ratios(
    model: nn.Module,
    stats: dict[str, ActStat],
    act_quantile: float = 0.999,
    weight_quantile: float = 0.999,
) -> None:
    for name, module in iter_quant_linears(model):
        if name in stats:
            module.state.act_clip_ratio = quantile_clip_ratio(stats[name].amax, act_quantile)
        module.state.weight_clip_ratio = quantile_clip_ratio(module.weight.detach(), weight_quantile)
        module.invalidate_cache()
