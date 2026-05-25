from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def _add_hifloat4_path_from_env() -> None:
    hifloat4_gpu_dir = os.getenv("HIFLOAT4_GPU_DIR")
    if not hifloat4_gpu_dir:
        return
    path = Path(hifloat4_gpu_dir).expanduser().resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@lru_cache(maxsize=None)
def _official_hif4_qtype_for_dim(qdim: int) -> Any:
    _add_hifloat4_path_from_env()
    try:
        from quant_cy import QType  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Official HiFloat4 CUDA backend is not available. Build "
            "global-computing-consortium/HiFloat4/hif4_gpu and either run from "
            "that directory or set HIFLOAT4_GPU_DIR=/path/to/HiFloat4/hif4_gpu. "
            "For the old repository fallback, set PTQ_WAN22_HIF4_BACKEND=fallback."
        ) from exc

    return QType("hifx4").dim(qdim)


def _official_hif4_qtype() -> Any:
    qdim = int(os.getenv("PTQ_WAN22_HIF4_QDIM", "-1"))
    return _official_hif4_qtype_for_dim(qdim)


def hif4_backend_name() -> str:
    return os.getenv("PTQ_WAN22_HIF4_BACKEND", "official").strip().lower()


def use_official_hif4() -> bool:
    backend = hif4_backend_name()
    if backend in {"official", "quant_cy", "hifloat4"}:
        return True
    if backend in {"fallback", "repo", "legacy"}:
        return False
    raise ValueError(
        "Unsupported PTQ_WAN22_HIF4_BACKEND. Use 'official' or 'fallback'."
    )


def official_hif4_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Run the official HiFloat4 CUDA pseudo-quantization kernel."""

    if not x.is_cuda:
        raise ValueError("Official HiFloat4 backend requires CUDA tensors.")

    _add_hifloat4_path_from_env()
    try:
        from quant_cy import quant_dequant_float  # type: ignore
    except ImportError as exc:
        _official_hif4_qtype_for_dim.cache_clear()
        raise ImportError(
            "Cannot import quant_cy.quant_dequant_float. Build HiFloat4/hif4_gpu "
            "with `bash build.sh`, then set HIFLOAT4_GPU_DIR to that directory."
        ) from exc

    force_fp32 = os.getenv("PTQ_WAN22_HIF4_FORCE_FP32", "0") == "1"

    q = quant_dequant_float(
        x.contiguous(),
        _official_hif4_qtype(),
        force_py=False,
        force_fp32=force_fp32,
    )
    return q.reshape_as(x)
