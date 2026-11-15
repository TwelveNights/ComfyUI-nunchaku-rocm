"""
What Triton needs on Windows beyond requirements.txt (nunchaku-rocm, triton-windows), checked when ComfyUI starts.
Nothing is installed or downloaded from here; what is missing is reported with the README section that covers it:

- the ROCm SDK devel package matching the installed ROCm wheels (HIP headers), from repo.radeon.com;
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
ROCM_DEVEL_URL = "https://repo.radeon.com/rocm/windows/rocm-rel-{v}/rocm_sdk_devel-{v}-py3-none-win_amd64.whl"
PYTHON_INCLUDE_LIBS = {
    (3, 10): "python_3.10.11_include_libs.zip",
    (3, 11): "python_3.11.9_include_libs.zip",
    (3, 12): "python_3.12.7_include_libs.zip",
    (3, 13): "python_3.13.2_include_libs.zip",
}
INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/{}"

MESSAGE: str | None = None  # set by check(); the DiT loader shows it instead of a Triton compile error


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _missing() -> list[str]:
    missing = []
    if importlib.util.find_spec("triton") is None:
        missing.append("triton-windows (listed in requirements.txt; reinstall the node pack's requirements)")
    rocm = _version("rocm-sdk-core")
    if rocm is not None and _version("rocm-sdk-devel") is None:
        wheel = ROCM_DEVEL_URL.format(v=rocm) if re.fullmatch(r"\d+\.\d+\.\d+", rocm) else f"version {rocm}"
        missing.append(f"the ROCm SDK devel package matching rocm-sdk-core {rocm}: {wheel}")
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
    MESSAGE = ("nunchaku-rocm needs a one-time setup on Windows (close ComfyUI first), see " + DOCS + ". Missing:\n  - "
               + "\n  - ".join(missing))
    logger.warning("[nunchaku-rocm] %s", MESSAGE)
    return MESSAGE
