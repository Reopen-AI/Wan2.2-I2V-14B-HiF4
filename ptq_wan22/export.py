from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .convert import iter_quant_linears
from .fp4_formats import FP4Config, block_fp4_quantize_to_packed


def _weight_for_export(module: nn.Module) -> torch.Tensor:
    weight = module.weight.detach()
    smooth_scale = module.state.smooth_scale
    if smooth_scale is None:
        return weight
    scale = smooth_scale.to(device=weight.device, dtype=weight.dtype)
    return weight * scale.reshape(1, -1)


def export_quantized_linear_weights(
    model: nn.Module,
    path: str | Path,
    *,
    include_bias: bool = True,
    scale_dtype: torch.dtype = torch.float16,
) -> None:
    """Export QuantLinear weights as packed 4-bit codes and block scales.

    This exporter mirrors the repository's fake-quant math. It folds
    SmoothQuant's input-channel scale into the weight before weight quantization
    and still saves the smooth scale because the runtime must divide activations
    by the same value before activation quantization.
    """

    export: dict[str, Any] = {
        "format_version": 1,
        "layout": "signed_fp4_nibbles_low_first",
        "linears": {},
    }

    for name, module in iter_quant_linears(model):
        cfg = FP4Config(
            fmt=module.weight_cfg.fmt,
            block_size=module.weight_cfg.block_size,
            eps=module.weight_cfg.eps,
            clip_ratio=module.state.weight_clip_ratio,
        )
        packed = block_fp4_quantize_to_packed(_weight_for_export(module), cfg)
        packed["scales"] = packed["scales"].to(scale_dtype)

        item: dict[str, Any] = {
            "weight": packed,
            "act_block_size": module.act_cfg.block_size,
            "act_clip_ratio": module.state.act_clip_ratio,
            "weight_clip_ratio": module.state.weight_clip_ratio,
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
        if module.state.smooth_scale is not None:
            item["smooth_scale"] = module.state.smooth_scale.detach().cpu().to(scale_dtype)
        else:
            item["smooth_scale"] = None
        if include_bias and module.bias is not None:
            item["bias"] = module.bias.detach().cpu()
        else:
            item["bias"] = None

        export["linears"][name] = item

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, out)


def export_dequantized_linear_weights(
    model: nn.Module,
    path: str | Path,
    *,
    include_bias: bool = True,
    weight_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
) -> None:
    """Export quant-dequant Linear weights plus runtime metadata.

    This is the format to use with the official HiFloat4 simulator backend,
    because the official package exposes quant-dequant kernels rather than a
    packed checkpoint layout. The saved `weight` tensor is already passed
    through `QuantLinear._qweight()`, so SmoothQuant's weight-side scale is
    folded into it. The activation-side `smooth_scale` is still saved and must
    be applied by any runtime that wants bitwise-equivalent behavior.
    """

    export: dict[str, Any] = {
        "format_version": 1,
        "layout": "quant_dequant_float_weights",
        "linears": {},
    }

    qdq_device = torch.device(device) if device is not None else None
    if qdq_device is None and torch.cuda.is_available():
        qdq_device = torch.device("cuda", torch.cuda.current_device())

    for name, module in iter_quant_linears(model):
        with torch.no_grad():
            orig_device = module.weight.device
            moved_for_qdq = qdq_device is not None and orig_device != qdq_device
            if moved_for_qdq:
                module.weight = nn.Parameter(
                    module.weight.detach().to(qdq_device),
                    requires_grad=False,
                )
                if module.bias is not None:
                    module.bias = nn.Parameter(
                        module.bias.detach().to(qdq_device),
                        requires_grad=False,
                    )
                module.invalidate_cache()
            try:
                qweight = module._qweight().detach().cpu().to(weight_dtype)
            finally:
                if moved_for_qdq:
                    module.weight = nn.Parameter(
                        module.weight.detach().to(orig_device),
                        requires_grad=False,
                    )
                    if module.bias is not None:
                        module.bias = nn.Parameter(
                            module.bias.detach().to(orig_device),
                            requires_grad=False,
                        )
                    module.invalidate_cache()
        item: dict[str, Any] = {
            "weight": qweight,
            "act_block_size": module.act_cfg.block_size,
            "act_clip_ratio": module.state.act_clip_ratio,
            "weight_block_size": module.weight_cfg.block_size,
            "weight_clip_ratio": module.state.weight_clip_ratio,
            "fmt": module.weight_cfg.fmt,
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
        if module.state.smooth_scale is not None:
            item["smooth_scale"] = module.state.smooth_scale.detach().cpu().to(weight_dtype)
        else:
            item["smooth_scale"] = None
        if include_bias and module.bias is not None:
            item["bias"] = module.bias.detach().cpu().to(weight_dtype)
        else:
            item["bias"] = None

        export["linears"][name] = item
        module.invalidate_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, out)


def load_dequantized_linear_weights(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str = "cpu",
    keep_export_dtype: bool = False,
) -> None:
    """Load a quant-dequant weight export into an already-converted model.

    The target model must already have `nn.Linear` modules replaced by
    `QuantLinear`. Loaded weights are already quant-dequanted, so this disables
    weight quantization for those modules. Activation quantization remains
    controlled by each module's `quantize_act` flag.
    """

    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("layout") != "quant_dequant_float_weights":
        raise ValueError(
            f"{path} is not a quant-dequant weight export. "
            "Use MODE=qdq when exporting official HiFloat4 weights."
        )

    exported = checkpoint["linears"]
    missing: list[str] = []
    for name, module in iter_quant_linears(model):
        if name not in exported:
            missing.append(name)
            continue

        item = exported[name]
        target_dtype = item["weight"].dtype if keep_export_dtype else module.weight.dtype
        weight = item["weight"].to(device=module.weight.device, dtype=target_dtype)
        module.weight = nn.Parameter(weight, requires_grad=False)

        if item.get("bias") is not None:
            bias_dtype = item["bias"].dtype if keep_export_dtype else weight.dtype
            bias = item["bias"].to(device=weight.device, dtype=bias_dtype)
            module.bias = nn.Parameter(bias, requires_grad=False)
        else:
            module.bias = None

        module.state.weight_clip_ratio = float(item["weight_clip_ratio"])
        module.state.act_clip_ratio = float(item["act_clip_ratio"])
        smooth_scale = item.get("smooth_scale")
        module.state.smooth_scale = smooth_scale if smooth_scale is None else smooth_scale.cpu()
        module.quantize_weight = False
        module.invalidate_cache()

    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{path} is missing {len(missing)} QuantLinear entries; first missing: {preview}"
        )
