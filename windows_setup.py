"""
What Triton needs on Windows beyond requirements.txt (nunchaku-rocm, triton-windows), checked when ComfyUI starts:

- the ROCm SDK devel package matching the installed ROCm wheels (HIP headers). It comes from repo.radeon.com, so pip
  cannot pick the matching version from requirements.txt; we print the one command that installs it.
- Python's ``include`` / ``libs`` folders, which the embedded Python of ComfyUI's portable build leaves out. These
  are plain files, downloaded once from the triton-windows project.
"""

import importlib.metadata as metadata
import importlib.util
import logging
import os
import re
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

DOCS = "https://github.com/TwelveNights/ComfyUI-nunchaku-rocm#install"
ROCM_DEVEL_URL = "https://repo.radeon.com/rocm/windows/rocm-rel-{v}/rocm_sdk_devel-{v}-py3-none-win_amd64.whl"
PYTHON_INCLUDE_LIBS = {
    (3, 10): "python_3.10.11_include_libs.zip",
    (3, 11): "python_3.11.9_include_libs.zip",
    (3, 12): "python_3.12.7_include_libs.zip",
    (3, 13): "python_3.13.2_include_libs.zip",
}
INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/{}"


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def missing_rocm_devel() -> str | None:
    """None if the ROCm SDK devel package is there (or not needed), else what to run to install it."""
    rocm = _version("rocm-sdk-core")
    if rocm is None or _version("rocm-sdk-devel") is not None:
        return None
    python = f'"{sys.executable}" -s -m pip install'
    if re.fullmatch(r"\d+\.\d+\.\d+", rocm):
        return f"{python} {ROCM_DEVEL_URL.format(v=rocm)}"
    return f"{python} rocm-sdk-devel=={rocm} from the same index as your ROCm wheels (rocm-sdk-core {rocm}), see {DOCS}"


def add_python_include_libs() -> None:
    """Embedded Pythons ship without include/ and libs/; Triton compiles a small launcher against both."""
    base = os.path.dirname(sysconfig.get_paths()["include"])
    lib = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    if os.path.exists(os.path.join(base, "include", "Python.h")) and os.path.exists(os.path.join(base, "libs", lib)):
        return
    name = PYTHON_INCLUDE_LIBS.get(sys.version_info[:2])
    if name is None:
        logger.warning("[nunchaku-rocm] no include/libs download for Python %s; Triton may fail to compile, see %s",
                       sys.version.split()[0], DOCS)
        return
    logger.info("[nunchaku-rocm] adding Python's include/ and libs/ to %s (%s, once)", base, name)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, name)
            urllib.request.urlretrieve(INCLUDE_LIBS_URL.format(name), path)
            with zipfile.ZipFile(path) as z:
                z.extractall(base)
    except Exception as e:  # noqa: BLE001 - report and let ComfyUI start
        logger.warning("[nunchaku-rocm] could not add Python's include/libs (%s), see %s", e, DOCS)


MESSAGE: str | None = None  # set by check(); the DiT loader shows it instead of a Triton compile error


def check() -> str | None:
    """Run at startup on Windows with a ROCm PyTorch. Returns a message if something has to be installed by hand."""
    global MESSAGE
    add_python_include_libs()
    problems = []
    if importlib.util.find_spec("triton") is None:
        problems.append(f'"{sys.executable}" -s -m pip install "triton-windows>=3.6.0.post26,<3.7"')
    devel = missing_rocm_devel()
    if devel:
        problems.append(devel)
    if not problems:
        return None
    MESSAGE = ("nunchaku-rocm needs a one-time setup on Windows. Close ComfyUI, run this in a terminal, then start "
               "ComfyUI again:\n  " + "\n  ".join(problems))
    logger.warning("[nunchaku-rocm] %s", MESSAGE)
    return MESSAGE
