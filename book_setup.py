"""Shared setup for the notebooks in this repo: PyTorch device and output paths.

Import this before ``torch`` in a notebook's first cell::

    from book_setup import DEVICE

``DEVICE`` is picked automatically -- CUDA (or ROCm) if present, then Apple
Silicon's MPS backend, then Intel XPU, then CPU -- so the same notebook runs
unmodified on an NVIDIA box, a MacBook, or a laptop with no GPU at all.

Two environment variables override the defaults:

``AI_BOOK_DEVICE``
    Force a device, e.g. ``AI_BOOK_DEVICE=cpu`` to check whether a bug is
    backend-specific.

``AI_BOOK_EXPORT_DIR``
    Where notebooks write figures and CSVs. Defaults to ``exports/`` beside this
    file.
"""

import os
import sys
import warnings
from pathlib import Path

__all__ = ["DEVICE", "EXPORT_DIR", "REPO_ROOT", "get_device", "export_path", "describe"]

REPO_ROOT = Path(__file__).resolve().parent

# Some operators still have no Metal kernel. This makes them fall back to the CPU
# instead of raising NotImplementedError. PyTorch reads it while `import torch`
# registers the MPS fallback, so setting it afterwards has no effect -- hence the
# warning below when torch got in first.
if sys.platform == "darwin":
    _fallback_preset = "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if "torch" in sys.modules and not _fallback_preset:
        warnings.warn(
            "book_setup was imported after torch, so PYTORCH_ENABLE_MPS_FALLBACK "
            "did not take effect. Operators with no MPS kernel will raise instead "
            "of falling back to the CPU. Import book_setup first, or set the "
            "variable in your shell.",
            RuntimeWarning,
            stacklevel=2,
        )

import torch


def get_device() -> str:
    """Best available PyTorch backend, or whatever ``AI_BOOK_DEVICE`` demands."""
    requested = os.environ.get("AI_BOOK_DEVICE")
    if requested:
        return requested

    # ROCm builds report themselves through the CUDA API, so this covers AMD too.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def export_path(*parts) -> Path:
    """Path under the export directory, with its parent directories created.

    ``plt.savefig`` and ``DataFrame.to_csv`` both refuse to create missing
    directories, so build write targets with this rather than by hand.
    """
    path = EXPORT_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def describe() -> str:
    """One-line summary of what the notebooks are about to run on."""
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        return f"{DEVICE} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}"
    if DEVICE == "mps":
        return f"mps (Apple Silicon GPU), torch {torch.__version__}"
    return f"{DEVICE}, torch {torch.__version__}"


DEVICE = get_device()
EXPORT_DIR = Path(os.environ.get("AI_BOOK_EXPORT_DIR", REPO_ROOT / "exports"))

# smalldiffusion builds an accelerate Accelerator internally, and that does its
# own device detection. Without this, AI_BOOK_DEVICE=cpu on a machine that has a
# GPU leaves the model on the GPU while the notebook puts its tensors on the CPU.
if DEVICE == "cpu":
    os.environ.setdefault("ACCELERATE_USE_CPU", "1")
