from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .hifloat4_backend import official_hif4_quant_dequant, use_official_hif4


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(x)
    return x + (rounded - x).detach()


@dataclass(frozen=True)
class FP4Config:
    """Configuration for block floating FP4 fake quantization.

    The code implements a practical MXFP4 E2M1 path and a HiF4-compatible
    fallback. If the official HiF4 kernels/simulator are available in your
    environment, wire them in at `quantize_hif4`.
    """

    fmt: str = "hif4"
    block_size: int = 64
    eps: float = 1e-8
    clip_ratio: float = 1.0


def _fp4_e2m1_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # E2M1 finite positive values used by common FP4/MXFP4 simulators.
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    return vals.to(dtype)


def _nearest_codebook(x_abs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map magnitudes to the nearest sorted codebook entry without NxK expand.

    This intentionally avoids `argmin(abs(x[..., None] - codebook))`, which can
    allocate tens of GB for Wan2.2 video activations.
    """
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    idx = torch.bucketize(x_abs.contiguous().reshape(-1), boundaries)
    return codebook[idx].reshape_as(x_abs)


def _nearest_codebook_indices(x_abs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    boundaries = (codebook[:-1] + codebook[1:]) * 0.5
    return torch.bucketize(x_abs.contiguous().reshape(-1), boundaries).reshape_as(x_abs)


def _pad_last_dim(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    rem = x.shape[-1] % block_size
    if rem == 0:
        return x, 0
    pad = block_size - rem
    return torch.nn.functional.pad(x, (0, pad)), pad


def block_fp4_quantize(x: torch.Tensor, cfg: FP4Config) -> torch.Tensor:
    """Fake-quantize `x` to block-scaled FP4 and dequantize back to `x.dtype`.

    `x` is grouped along the last dimension, which matches Linear input-channel
    grouping for weights and per-token grouping for activations.
    """

    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive")

    fmt = cfg.fmt.lower()
    if fmt == "hif4" and use_official_hif4():
        orig_dtype = x.dtype
        q = official_hif4_quant_dequant(x.contiguous())
        return q.to(orig_dtype)

    orig_dtype = x.dtype
    work = x.float()
    work, pad = _pad_last_dim(work, cfg.block_size)
    grouped = work.reshape(*work.shape[:-1], -1, cfg.block_size)

    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(cfg.eps)
    if cfg.clip_ratio < 1.0:
        amax = amax * cfg.clip_ratio

    if fmt == "mxfp4":
        q = _quantize_mxfp4_grouped(grouped, amax)
    elif fmt == "hif4":
        q = _quantize_hif4_grouped(grouped, amax)
    else:
        raise ValueError(f"Unsupported FP4 format: {cfg.fmt}")

    q = q.reshape_as(work)
    if pad:
        q = q[..., :-pad]
    return q.to(orig_dtype)


def block_fp4_quantize_to_packed(x: torch.Tensor, cfg: FP4Config) -> dict[str, torch.Tensor | tuple[int, ...] | int | str]:
    """Quantize `x` to packed signed FP4 codes plus per-block scales.

    The packed nibble layout is a reference export format for this repository:
    each 4-bit code stores sign in bit 3 and a 3-bit magnitude-codebook index in
    bits 0..2. Two codes are packed into one uint8, with the first value in the
    low nibble. Official HiF4/MXFP4 runtimes may require a different metadata
    layout, but this contains the information needed to reconstruct the current
    fake-quantized weights.
    """

    if cfg.fmt.lower() == "hif4" and use_official_hif4():
        raise NotImplementedError(
            "The official HiFloat4 package exposes quant-dequant simulation, not "
            "a packed weight export layout. Set PTQ_WAN22_HIF4_BACKEND=fallback "
            "only if you want this repository's old reference packed format."
        )

    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive")

    work = x.detach().float()
    orig_shape = tuple(work.shape)
    work, pad = _pad_last_dim(work, cfg.block_size)
    grouped = work.reshape(*work.shape[:-1], -1, cfg.block_size)

    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(cfg.eps)
    if cfg.clip_ratio < 1.0:
        amax = amax * cfg.clip_ratio

    fmt = cfg.fmt.lower()
    if fmt == "mxfp4":
        codebook = _fp4_e2m1_codebook(grouped.device, grouped.dtype)
    elif fmt == "hif4":
        codebook = torch.tensor(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            device=grouped.device,
            dtype=grouped.dtype,
        )
    else:
        raise ValueError(f"Unsupported FP4 format: {cfg.fmt}")

    max_normal = codebook[-1]
    scales = (amax / max_normal).clamp_min(cfg.eps)
    normalized = torch.clamp(grouped / scales, -max_normal, max_normal)
    mag_idx = _nearest_codebook_indices(normalized.abs(), codebook).to(torch.uint8)
    sign = (normalized < 0).to(torch.uint8) << 3
    codes = (mag_idx | sign).reshape(-1)

    if codes.numel() % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    packed = (codes[0::2] | (codes[1::2] << 4)).contiguous()

    return {
        "packed": packed.cpu(),
        "scales": scales.squeeze(-1).cpu(),
        "orig_shape": orig_shape,
        "padded_last_dim": work.shape[-1],
        "pad": pad,
        "block_size": cfg.block_size,
        "fmt": fmt,
    }


def _quantize_mxfp4_grouped(grouped: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    # Scale maps the largest magnitude in the block to the largest E2M1 value.
    codebook = _fp4_e2m1_codebook(grouped.device, grouped.dtype)
    max_normal = codebook[-1]
    scale = (amax / max_normal).clamp_min(1e-8)
    normalized = torch.clamp(grouped / scale, -max_normal, max_normal)
    q_abs = _nearest_codebook(normalized.abs(), codebook)
    return torch.sign(normalized) * q_abs * scale


def _quantize_hif4_grouped(grouped: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """HiF4-style fallback.

    Official HiF4 uses hierarchical shared scaling metadata. For search and
    ablation this fallback behaves like a denser non-uniform FP4 codebook with
    block scaling. Replace this function with the official simulator call when
    submitting to the challenge.
    """

    codebook = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        device=grouped.device,
        dtype=grouped.dtype,
    )
    max_normal = codebook[-1]
    scale = (amax / max_normal).clamp_min(1e-8)
    normalized = torch.clamp(grouped / scale, -max_normal, max_normal)
    q_abs = _nearest_codebook(normalized.abs(), codebook)
    return torch.sign(normalized) * q_abs * scale


def quantile_clip_ratio(
    x: torch.Tensor,
    quantile: float = 0.999,
    max_samples: int = 1_000_000,
) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    abs_x = x.detach().float().abs().flatten()
    if abs_x.numel() == 0:
        return 1.0

    # `torch.quantile` can reject very large tensors. For PTQ clipping, an
    # evenly-strided deterministic sample is stable enough and keeps memory
    # bounded for Wan2.2's large Linear weights.
    if abs_x.numel() > max_samples:
        stride = math.ceil(abs_x.numel() / max_samples)
        abs_x = abs_x[::stride]

    q = torch.quantile(abs_x, quantile)
    mx = abs_x.max().clamp_min(1e-8)
    return float((q / mx).clamp(max=1.0).item())


def next_power_of_two(x: float) -> float:
    if x <= 0:
        return 1.0
    return 2.0 ** math.ceil(math.log2(x))
