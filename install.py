"""
Run by ComfyUI-Manager after it installs this node pack (or by hand: ``<ComfyUI's python> install.py``).

Installs nunchaku-rocm into ComfyUI's Python environment on top of the ROCm PyTorch that is already there. The
installed torch / torchvision / torchaudio / triton / rocm packages are pinned to their current versions, so pip can
never swap them for CUDA builds. On Windows it also adds what Triton needs to compile there:

- ``triton-windows``,
- the ROCm SDK devel package matching the installed ROCm wheels (HIP headers),
- Python's ``include`` / ``libs`` folders, which ComfyUI's portable (embedded) Python leaves out.

Nothing is installed unless PyTorch is a ROCm build.
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

NUNCHAKU_ROCM = "nunchaku-rocm==1.3.0"
TRITON_WINDOWS = "triton-windows>=3.6.0.post26,<3.7"
ROCM_WINDOWS = "https://repo.radeon.com/rocm/windows/rocm-rel-{v}/rocm_sdk_devel-{v}-py3-none-win_amd64.whl"
# Python headers and import libraries for embedded Pythons, from the triton-windows project
PYTHON_INCLUDE_LIBS = {
    (3, 10): "python_3.10.11_include_libs.zip",
    (3, 11): "python_3.11.9_include_libs.zip",
    (3, 12): "python_3.12.7_include_libs.zip",
    (3, 13): "python_3.13.2_include_libs.zip",
}
INCLUDE_LIBS_URL = "https://github.com/woct0rdho/triton-windows/releases/download/v3.0.0-windows.post1/{}"
DOCS = "https://github.com/TwelveNights/ComfyUI-nunchaku-rocm#install"
PINNED = re.compile(r"(torch|torchvision|torchaudio|triton|pytorch-triton|rocm)", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[ComfyUI-nunchaku-rocm] {msg}", flush=True)


def version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def keep_installed(installed: str | None) -> bool:
    """An editable checkout or a release at least as new as the one this node pack needs is left alone."""
    if installed is None:
        return False
    direct_url = metadata.distribution("nunchaku-rocm").read_text("direct_url.json") or ""
    if '"editable": true' in direct_url.replace("'", '"'):
        return True
    try:
        from packaging.version import Version

        return Version(installed) >= Version(NUNCHAKU_ROCM.split("==")[1])
    except Exception:  # noqa: BLE001
        return installed == NUNCHAKU_ROCM.split("==")[1]


def pip_install(args: list[str], constraints: str) -> None:
    cmd = [sys.executable, "-s", "-m", "pip", "install", "-c", constraints, *args]
    log(" ".join(cmd))
    subprocess.check_call(cmd)


def add_python_include_libs() -> None:
    """Embedded Pythons (ComfyUI portable) ship without include/ and libs/; Triton compiles against both."""
    base = os.path.dirname(sysconfig.get_paths()["include"])
    lib = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    if os.path.exists(os.path.join(base, "include", "Python.h")) and os.path.exists(os.path.join(base, "libs", lib)):
        return
    name = PYTHON_INCLUDE_LIBS.get(sys.version_info[:2])
    if name is None:
        log(f"no Python include/libs download for Python {sys.version.split()[0]}; Triton may fail to compile")
        return
    log(f"adding Python's include/ and libs/ to {base} (from {name})")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        urllib.request.urlretrieve(INCLUDE_LIBS_URL.format(name), path)
        with zipfile.ZipFile(path) as z:
            z.extractall(base)


def main() -> int:
    try:
        import torch
    except ImportError:
        log(f"PyTorch is not installed in this environment; install a ROCm PyTorch first, see {DOCS}")
        return 0
    if torch.version.hip is None:
        log(f"PyTorch {torch.__version__} is not a ROCm build; nothing installed. This node pack runs on AMD GPUs only.")
        return 0

    installed = version("nunchaku-rocm")
    if installed is None and version("nunchaku") is not None:
        log("another 'nunchaku' package is installed; leaving it alone. Install nunchaku-rocm by hand, see " + DOCS)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        constraints = os.path.join(tmp, "constraints.txt")
        with open(constraints, "w") as f:
            for dist in metadata.distributions():
                name = dist.metadata["Name"] or ""
                if PINNED.match(name):
                    f.write(f"{name}=={dist.version}\n")

        packages = [] if keep_installed(installed) else [NUNCHAKU_ROCM]
        if os.name == "nt":
            if version("triton-windows") is None:
                packages.append(TRITON_WINDOWS)
            rocm = version("rocm-sdk-core")
            if rocm is not None and version("rocm-sdk-devel") is None:
                if re.fullmatch(r"7\.\d+\.\d+", rocm):
                    packages.append(ROCM_WINDOWS.format(v=rocm))
                else:
                    log(f"install the ROCm SDK devel package matching rocm-sdk-core {rocm} by hand, see {DOCS}")
        try:
            if packages:
                pip_install(packages, constraints)
            if os.name == "nt":
                add_python_include_libs()
        except Exception as e:  # noqa: BLE001 - report and let ComfyUI start
            log(f"install failed ({e}); see {DOCS}")
            return 1

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
