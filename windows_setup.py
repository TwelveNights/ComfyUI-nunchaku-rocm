"""
What Triton needs on Windows beyond requirements.txt (nunchaku-rocm), checked when ComfyUI starts.
Nothing is installed or downloaded from here (install.py does that); what is missing is reported:

- triton-windows in the release series that matches the installed PyTorch;
- the ROCm SDK devel package matching the installed ROCm wheels (HIP headers), from AMD's package servers;
- Python's ``include`` / ``libs`` folders, which the embedded Python of ComfyUI's portable build leaves out.
"""

import importlib.metadata as metadata
import importlib.util
import logging
import os
import re
import sys
import sysconfig

logger = logging.getLogger(__name__)

DOCS = "https://github.com/TwelveNights/ComfyUI-nunchaku-rocm#install (step 2, Windows only)"
# ROCm 7.x Windows wheels are on repo.radeon.com; ROCm 10 and later are on AMD's whl-next index.
ROCM_DEVEL_URL = "https://repo.radeon.com/rocm/windows/rocm-rel-{v}/rocm_sdk_devel-{v}-py3-none-win_amd64.whl"
ROCM_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"
PYTHON_INCLUDE_LIBS = {
    (3, 10): "python_3.10.11_include_libs.zip",
    (3, 11): "python_3.11.9_include_libs.zip",
    (3, 12): "python_3.12.7_include_libs.zip",
    (3, 13): "python_3.13.2_include_libs.zip",
}
INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/{}"

# triton-windows release series for each PyTorch minor version: the Triton version PyTorch's own Linux wheels pin,
# except that before 2.12 it is 3.6, the first triton-windows release with the AMD backend (3.6.0.post26).
# torch.compile needs the matching series; newer PyTorch gets the newest series.
TRITON_FOR_TORCH = [
    ((2, 11), ">=3.6.0.post26,<3.7"),
    ((2, 12), ">=3.7.0,<3.7.1"),
    ((2, 13), ">=3.7.1,<3.8"),
    ((2, 14), ">=3.8,<3.9"),
    ((2, 15), ">=3.9,<3.10"),
]


def triton_windows_requirement(torch_version: str) -> str:
    """The triton-windows requirement matching a PyTorch version such as ``2.9.1+rocm7.2.1``."""
    m = re.match(r"(\d+)\.(\d+)", torch_version)
    minor = (int(m.group(1)), int(m.group(2))) if m else (99, 0)
    for upto, spec in TRITON_FOR_TORCH:
        if minor <= upto:
            return "triton-windows" + spec
    return "triton-windows>=3.9"


def rocm_devel_pip_args(rocm_version: str) -> list[str] | None:
    """pip install arguments for the rocm-sdk-devel matching a rocm-sdk-core release version, or None."""
    m = re.fullmatch(r"(\d+)\.\d+\.\d+", rocm_version)
    if m is None:
        return None
    if int(m.group(1)) < 10:
        return [ROCM_DEVEL_URL.format(v=rocm_version)]
    return ["--index-url", ROCM_INDEX_URL, f"rocm-sdk-devel=={rocm_version}"]


def _torch_version() -> str:
    return _version("torch") or ""


MESSAGE: str | None = None  # set by check(); the DiT loader shows it instead of a Triton compile error


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _missing() -> list[str]:
    missing = []
    if importlib.util.find_spec("triton") is None:
        missing.append(f"{triton_windows_requirement(_torch_version())} (matching PyTorch {_torch_version()})")
    rocm = _version("rocm-sdk-core")
    if rocm is not None and _version("rocm-sdk-devel") is None:
        args = rocm_devel_pip_args(rocm)
        source = "pip install " + " ".join(args) if args else f"version {rocm}"
        missing.append(f"the ROCm SDK devel package matching rocm-sdk-core {rocm}: {source}")
    base = os.path.dirname(sysconfig.get_paths()["include"])
    lib = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    if not (os.path.exists(os.path.join(base, "include", "Python.h")) and os.path.exists(os.path.join(base, "libs", lib))):
        name = PYTHON_INCLUDE_LIBS.get(sys.version_info[:2], f"the include/libs zip for Python {sys.version.split()[0]}")
        url = INCLUDE_LIBS_URL.format(name) if name.endswith(".zip") else name
        missing.append(f"Python's include/ and libs/ folders in {base}: unpack {url} there")
    return missing


def check() -> str | None:
    """Run at startup on Windows with a ROCm PyTorch. Returns a message if something has to be set up by hand."""
    global MESSAGE
    missing = _missing()
    if not missing:
        return None
    MESSAGE = ("nunchaku-rocm needs a one-time setup on Windows: close ComfyUI and run this node pack's install.py with "
               f'ComfyUI\'s Python ("{sys.executable}" install.py in {os.path.dirname(os.path.abspath(__file__))}), '
               "or see " + DOCS + ". Missing:\n  - " + "\n  - ".join(missing))
    logger.warning("[nunchaku-rocm] %s", MESSAGE)
    return MESSAGE
