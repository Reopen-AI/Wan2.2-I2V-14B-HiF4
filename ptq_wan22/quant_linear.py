from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .fp4_formats import FP4Config, block_fp4_quantize


@dataclass
class LinearQuantState:
    name: str
    weight_clip_ratio: float = 1.0
    act_clip_ratio: float = 1.0
    smooth_scale: Optional[torch.Tensor] = None
    enabled: bool = True


class QuantLinear(nn.Module):
    """Drop-in fake-quantized Linear.

    This module intentionally keeps weights in torch tensors, so you can use it
    for calibration, sensitivity search, and challenge-side simulator runs. For
    final high-performance inference, export the quantized weights into the
    official HiF4/MXFP4 runtime kernels.
    """

    def __init__(
        self,
        src: nn.Linear,
        name: str,
        weight_cfg: FP4Config,
        act_cfg: FP4Config,
        quantize_weight: bool = True,
        quantize_act: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = src.in_features
        self.out_features = src.out_features
        src.weight.requires_grad_(False)
        self.weight = src.weight
        if src.bias is None:
            self.bias = None
        else:
            src.bias.requires_grad_(False)
            self.bias = src.bias
        self.weight_cfg = weight_cfg
        self.act_cfg = act_cfg
        self.quantize_weight = quantize_weight
        self.quantize_act = quantize_act
        self.state = LinearQuantState(name=name)
        self.register_buffer("_cached_qweight", torch.empty(0), persistent=False)
        self._cache_key: Optional[tuple] = None

    @classmethod
    def from_linear(
        cls,
        src: nn.Linear,
        name: str,
        weight_cfg: FP4Config,
        act_cfg: FP4Config,
    ) -> "QuantLinear":
        return cls(src, name=name, weight_cfg=weight_cfg, act_cfg=act_cfg)

    def invalidate_cache(self) -> None:
        self._cached_qweight = torch.empty(0, device=self.weight.device)
        self._cache_key = None

    def _qweight(self) -> torch.Tensor:
        if not self.quantize_weight or not self.state.enabled:
            return self.weight

        key = (
            self.weight.data_ptr(),
            self.weight.dtype,
            self.weight.device,
            self.weight_cfg.fmt,
            self.weight_cfg.block_size,
            self.state.weight_clip_ratio,
            None if self.state.smooth_scale is None else self.state.smooth_scale.data_ptr(),
        )
        if self._cache_key == key and self._cached_qweight.numel():
            return self._cached_qweight

        weight = self.weight
        if self.state.smooth_scale is not None:
            scale = self.state.smooth_scale.to(device=weight.device, dtype=weight.dtype)
            weight = weight * scale.reshape(1, -1)
        cfg = FP4Config(
            fmt=self.weight_cfg.fmt,
            block_size=self.weight_cfg.block_size,
            eps=self.weight_cfg.eps,
            clip_ratio=self.state.weight_clip_ratio,
        )
        self._cached_qweight = block_fp4_quantize(weight, cfg)
        self._cache_key = key
        return self._cached_qweight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.state.smooth_scale is not None:
            scale = self.state.smooth_scale.to(device=x.device, dtype=x.dtype)
            x = x / scale.reshape(*([1] * (x.ndim - 1)), -1).clamp_min(1e-8)

        if self.quantize_act and self.state.enabled:
            cfg = FP4Config(
                fmt=self.act_cfg.fmt,
                block_size=self.act_cfg.block_size,
                eps=self.act_cfg.eps,
                clip_ratio=self.state.act_clip_ratio,
            )
            x = block_fp4_quantize(x, cfg)

        return F.linear(x, self._qweight(), self.bias)
