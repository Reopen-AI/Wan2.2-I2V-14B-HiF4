from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch import nn

from .fp4_formats import FP4Config
from .quant_linear import QuantLinear


@dataclass
class QuantConfig:
    fmt: str = "hif4"
    weight_block_size: int = 64
    act_block_size: int = 64
    keep_fp_regex: list[str] = field(default_factory=list)
    quantize_weight: bool = True
    quantize_act: bool = True

    def weight_fp4(self) -> FP4Config:
        return FP4Config(fmt=self.fmt, block_size=self.weight_block_size)

    def act_fp4(self) -> FP4Config:
        return FP4Config(fmt=self.fmt, block_size=self.act_block_size)


def _should_keep_fp(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def first_last_block_keep_fp_regex(model: nn.Module) -> list[str]:
    """Build regexes that keep the first and last transformer blocks in BF16."""

    block_ids_by_prefix: dict[str, set[int]] = {}
    pattern = re.compile(r"^(?P<prefix>.*?blocks)\.(?P<idx>\d+)(?:\.|$)")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        match = pattern.match(name)
        if match is None:
            continue
        block_ids_by_prefix.setdefault(match.group("prefix"), set()).add(int(match.group("idx")))

    keep: list[str] = []
    for prefix, block_ids in sorted(block_ids_by_prefix.items()):
        if not block_ids:
            continue
        first = min(block_ids)
        last = max(block_ids)
        if first == last:
            block_expr = str(first)
        else:
            block_expr = f"{first}|{last}"
        keep.append(rf"^{re.escape(prefix)}\.({block_expr})(\.|$)")
    return keep


def _first_last_linear_names(model: nn.Module) -> list[str]:
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not names:
        return []
    if len(names) == 1:
        return [names[0]]
    return [names[0], names[-1]]


def first_last_linear_keep_fp_regex(model: nn.Module) -> list[str]:
    """Build exact regexes that keep each expert's first/last Linear in BF16."""

    expert_names = [
        name
        for name in ("low_noise_model", "high_noise_model")
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Module)
    ]
    if not expert_names:
        return [rf"^{re.escape(name)}$" for name in _first_last_linear_names(model)]

    keep: list[str] = []
    for expert_name in expert_names:
        expert = getattr(model, expert_name)
        for linear_name in _first_last_linear_names(expert):
            keep.append(rf"^{re.escape(expert_name + '.' + linear_name)}$")
    return keep


def _get_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def convert_linear_modules(model: nn.Module, cfg: QuantConfig) -> dict[str, nn.Linear]:
    """Replace Linear modules with QuantLinear and return original modules.

    `keep_fp_regex` is the hook for challenge rules: put the two HiF4 or five
    MXFP4 high-precision blocks here after sensitivity ranking.
    """

    originals: dict[str, nn.Linear] = {}
    replacements: list[tuple[str, QuantLinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _should_keep_fp(name, cfg.keep_fp_regex):
            qmod = QuantLinear.from_linear(
                module,
                name=name,
                weight_cfg=cfg.weight_fp4(),
                act_cfg=cfg.act_fp4(),
            )
            qmod.quantize_weight = cfg.quantize_weight
            qmod.quantize_act = cfg.quantize_act
            originals[name] = module
            replacements.append((name, qmod))

    for name, qmod in replacements:
        parent, child_name = _get_parent(model, name)
        setattr(parent, child_name, qmod)

    return originals


def restore_fp_modules(model: nn.Module, originals: dict[str, nn.Linear]) -> None:
    for name, module in originals.items():
        parent, child_name = _get_parent(model, name)
        setattr(parent, child_name, module)


def iter_quant_linears(model: nn.Module) -> Iterable[tuple[str, QuantLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            yield name, module


def save_quant_state(model: nn.Module, path: str) -> None:
    state = {}
    for name, module in iter_quant_linears(model):
        state[name] = {
            "weight_clip_ratio": module.state.weight_clip_ratio,
            "act_clip_ratio": module.state.act_clip_ratio,
            "smooth_scale": module.state.smooth_scale.detach().cpu()
            if module.state.smooth_scale is not None
            else None,
        }
    torch.save(state, path)


def load_quant_state(model: nn.Module, path: str, map_location: str = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    for name, module in iter_quant_linears(model):
        if name not in state:
            continue
        item = state[name]
        module.state.weight_clip_ratio = float(item["weight_clip_ratio"])
        module.state.act_clip_ratio = float(item["act_clip_ratio"])
        module.state.smooth_scale = item["smooth_scale"]
        module.invalidate_cache()
