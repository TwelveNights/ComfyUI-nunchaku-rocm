"""
ComfyUI nodes for Nunchaku SVDQuant (4-bit) models on AMD GPUs, running on nunchaku-rocm.

Needs nunchaku-rocm installed in ComfyUI's Python environment: https://github.com/TwelveNights/nunchaku-rocm
"""

import importlib.util
import os
import sys

if importlib.util.find_spec("nunchaku") is None:
    raise ImportError(
        "ComfyUI-nunchaku-rocm needs nunchaku-rocm in ComfyUI's Python environment, installed on top of a ROCm "
        "PyTorch: see https://github.com/TwelveNights/nunchaku-rocm#install"
    )

# select the Triton backend before nunchaku is first imported; NUNCHAKU_BACKEND=cuda can still override
os.environ.setdefault("NUNCHAKU_BACKEND", "triton")


def _use_rocm_sdk() -> None:
    """Windows ROCm wheels: point Triton's compiler at the ROCm SDK (HIP headers, clang), unless already set up."""
    if os.environ.get("ROCM_HOME") or importlib.util.find_spec("rocm_sdk") is None:
        return
    try:
        from rocm_sdk import _devel

        root = str(_devel.get_devel_root())
    except Exception:  # noqa: BLE001 - the devel package is missing; install.py reports how to add it
        return
    os.environ["ROCM_HOME"] = root
    os.environ.setdefault("HIP_PATH", root)
    os.environ["PATH"] = os.pathsep.join(
        [os.path.join(root, "bin"), os.path.join(root, "lib", "llvm", "bin"), os.environ.get("PATH", "")]
    )
    # ComfyUI may have imported Triton already, and its AMD driver reads the ROCm location only at import
    driver = sys.modules.get("triton.backends.amd.driver")
    include = os.path.join(root, "include")
    if driver is not None and include not in getattr(driver, "include_dirs", [include]):
        driver.include_dirs.append(include)


if os.name == "nt":
    _use_rocm_sdk()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
