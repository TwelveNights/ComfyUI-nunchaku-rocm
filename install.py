"""
Run by ComfyUI-Manager after it installs this node pack and its requirements.txt (or by hand:
``<ComfyUI's python> install.py``).

Windows with a ROCm PyTorch only: adds what Triton needs there and requirements.txt cannot express,

- the ROCm SDK devel package matching the installed ROCm wheels (HIP headers), from repo.radeon.com, installed with
  the installed torch / torchvision / torchaudio / triton / rocm packages pinned so pip cannot replace them;
- Python's ``include`` / ``libs`` folders, which ComfyUI's portable (embedded) Python leaves out.

Does nothing on Linux, without a ROCm PyTorch, or when both are already there. If it cannot do its job, ComfyUI's
console says what is missing at the next start (see windows_setup.py).
"""

import importlib.metadata as metadata
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the portable's embedded Python skips the script dir
from windows_setup import INCLUDE_LIBS_URL, PYTHON_INCLUDE_LIBS, ROCM_DEVEL_URL  # noqa: E402

PINNED = re.compile(r"(torch|torchvision|torchaudio|triton|pytorch-triton|rocm)", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[ComfyUI-nunchaku-rocm] {msg}", flush=True)


def version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def add_rocm_devel() -> None:
    rocm = version("rocm-sdk-core")
    if rocm is None or version("rocm-sdk-devel") is not None:
        return
    if not re.fullmatch(r"\d+\.\d+\.\d+", rocm):
        log(f"install the ROCm SDK devel package matching rocm-sdk-core {rocm} by hand (README, install step 2)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        constraints = os.path.join(tmp, "constraints.txt")
        with open(constraints, "w") as f:
            for dist in metadata.distributions():
                name = dist.metadata["Name"] or ""
                if PINNED.match(name):
                    f.write(f"{name}=={dist.version}\n")
        cmd = [sys.executable, "-s", "-m", "pip", "install", "-c", constraints, ROCM_DEVEL_URL.format(v=rocm)]
        log(" ".join(cmd))
        subprocess.check_call(cmd)


def add_python_include_libs() -> None:
    base = os.path.dirname(sysconfig.get_paths()["include"])
    lib = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    if os.path.exists(os.path.join(base, "include", "Python.h")) and os.path.exists(os.path.join(base, "libs", lib)):
        return
    name = PYTHON_INCLUDE_LIBS.get(sys.version_info[:2])
    if name is None:
        log(f"no include/libs download known for Python {sys.version.split()[0]}; see README, install step 2")
        return
    log(f"adding Python's include/ and libs/ to {base} (from {name})")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        urllib.request.urlretrieve(INCLUDE_LIBS_URL.format(name), path)
        with zipfile.ZipFile(path) as z:
            z.extractall(base)


def main() -> int:
    if os.name != "nt":
        return 0
    try:
        import torch
    except ImportError:
        return 0
    if not torch.version.hip:
        return 0
    try:
        add_rocm_devel()
        add_python_include_libs()
    except Exception as e:  # noqa: BLE001 - report and let ComfyUI start; windows_setup.py lists what is missing
        log(f"Windows setup failed ({e}); see README, install step 2")
        return 1
    log("Windows setup done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
