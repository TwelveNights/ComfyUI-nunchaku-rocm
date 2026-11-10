"""
ComfyUI nodes for Nunchaku SVDQuant (4-bit) models on AMD GPUs, running on nunchaku-rocm.

Needs nunchaku-rocm installed in ComfyUI's Python environment: https://github.com/TwelveNights/nunchaku-rocm
"""

import importlib.util
import os

if importlib.util.find_spec("nunchaku") is None:
    raise ImportError(
        "ComfyUI-nunchaku-rocm needs nunchaku-rocm in ComfyUI's Python environment, installed on top of a ROCm "
        "PyTorch: see https://github.com/TwelveNights/nunchaku-rocm#install"
    )

# select the Triton backend before nunchaku is first imported; NUNCHAKU_BACKEND=cuda can still override
os.environ.setdefault("NUNCHAKU_BACKEND", "triton")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
