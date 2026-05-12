# SPDX-FileCopyrightText: 2025-2026 Olgierd Nowakowski
# SPDX-License-Identifier: Apache-2.0

"""Optional GPU acceleration for array operations.

When a compatible GPU runtime is available (NVIDIA CUDA via CuPy, or AMD ROCm
via CuPy-ROCm) the heavy numpy array operations in the loaders are transparently
replaced with GPU equivalents for a significant speedup on large builds.

Install one of the following to enable GPU support:

    # NVIDIA CUDA 12.x
    pip install cupy-cuda12x

    # NVIDIA CUDA 11.x
    pip install cupy-cuda11x

    # AMD ROCm 5.0
    pip install cupy-rocm-5-0

Usage::

    from obfviewer.gpu import xp, using_gpu, gpu_info

    # xp is cupy if a GPU is available, numpy otherwise — use it everywhere
    arr = xp.array([1.0, 2.0, 3.0])
    result = xp.sqrt(arr)          # runs on GPU if available

    if using_gpu:
        print(f"GPU: {gpu_info()}")
"""

from __future__ import annotations

import os
import warnings

import numpy as _np


def _detect_cupy():
    """Attempt to import and initialise CuPy.

    Returns the cupy module on success, or None if unavailable.
    Silences import noise so a missing CuPy is never an error.
    """
    # Respect an explicit opt-out environment variable
    if os.environ.get("OBFVIEWER_NO_GPU", "").lower() in {"1", "true", "yes"}:
        return None

    try:
        import cupy as cp  # noqa: F401

        # Quick smoke-test: allocate a tiny array to confirm the runtime works.
        # AttributeError means the generic 'cupy' stub is installed without a
        # real backend wheel (e.g. cupy-cuda12x) — treat that silently as "not
        # installed" rather than alarming the user.
        if not hasattr(cp, "array"):
            return None

        cp.array([0], dtype=cp.float32)
        return cp
    except ImportError:
        # CuPy not installed at all — fall back to numpy silently.
        return None
    except AttributeError:
        # Stub package installed without a real CUDA/ROCm backend.
        return None
    except Exception as exc:
        # CuPy is properly installed but the GPU runtime itself failed
        # (e.g. driver version mismatch, no device found).
        warnings.warn(
            f"GPU runtime detected but failed to initialise: {exc}. "
            "Falling back to CPU (numpy). "
            "Set OBFVIEWER_NO_GPU=1 to suppress this warning.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


_cupy = _detect_cupy()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: The array namespace in use — cupy if a GPU is available, numpy otherwise.
#: Drop-in replacement: use ``xp.array(...)`` instead of ``np.array(...)`` in
#: performance-critical paths and you get GPU acceleration for free.
xp = _cupy if _cupy is not None else _np

#: True when GPU acceleration is active.
using_gpu: bool = _cupy is not None


def gpu_info() -> str:
    """Return a human-readable string describing the active GPU(s).

    Returns:
        Description string, e.g.  ``"NVIDIA GeForce RTX 3060 (CUDA 12.2)"``
        or ``"No GPU (CPU/numpy mode)"`` when running on CPU.
    """
    if not using_gpu or _cupy is None:
        return "No GPU (CPU/numpy mode)"

    try:
        import cupy as cp

        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props["name"].decode("utf-8") if isinstance(props["name"], bytes) else props["name"]
        cuda_ver = ""
        try:
            runtime_ver = cp.cuda.runtime.runtimeGetVersion()
            major, minor = divmod(runtime_ver, 1000)
            cuda_ver = f"CUDA {major}.{minor // 10}"
        except Exception:
            cuda_ver = "CUDA (version unknown)"

        mem_total = device.mem_info[1] / 1024**3
        return f"{name} — {mem_total:.1f} GB VRAM, {cuda_ver}"
    except Exception as exc:
        return f"GPU available (details unavailable: {exc})"


def to_numpy(arr) -> _np.ndarray:
    """Convert an array to numpy regardless of whether it is a cupy or numpy array.

    Args:
        arr: A numpy or cupy array.

    Returns:
        A numpy ndarray.
    """
    if using_gpu and _cupy is not None:
        try:
            return _cupy.asnumpy(arr)
        except Exception:
            pass
    return _np.asarray(arr)
