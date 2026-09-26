# Neural denoisers (RNNoise, DeepFilterNet) downloaded on first use; Krisp is enterprise-only
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

RNNOISE_URL = ("https://raw.githubusercontent.com/GregorR/rnnoise-models/master/"
               "somnolent-hogwash-2018-09-01/sh.rnnn")
DF_VERSION = "0.5.6"
DF_URL = f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{DF_VERSION}/deep-filter-{DF_VERSION}-"
# Keep some residual noise: fully "clean" output has artifacts that raise ASR error rates.
DF_ATTENUATION_DB = 20


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else None
    d = Path(base or Path.home() / ".cache") / "minutesman"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dst: Path) -> Path:
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dst)
    return dst


def rnnoise_model() -> Path:
    return _download(RNNOISE_URL, cache_dir() / "rnnoise-sh.rnnn")


def deepfilter_exe() -> Path:
    found = shutil.which("deep-filter")
    if found:
        return Path(found)
    machine = platform.machine().lower()
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    target = {
        "linux": f"{arch}-unknown-linux-musl" if arch == "x86_64" else f"{arch}-unknown-linux-gnu",
        "darwin": f"{arch}-apple-darwin",
        "win32": "x86_64-pc-windows-msvc.exe",
    }.get(sys.platform)
    if target is None:
        raise RuntimeError(f"No DeepFilterNet binary for {sys.platform}; install deep-filter on PATH")
    exe = _download(DF_URL + target, cache_dir() / f"deep-filter-{DF_VERSION}-{target}")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


# Run DeepFilterNet on a 48 kHz mono WAV with delay compensation
def deepfilter(src_48k: Path, dst: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [str(deepfilter_exe()), "-D", "-a", str(DF_ATTENUATION_DB), "-o", tmp, str(src_48k)],
            capture_output=True, text=True, errors="replace",
        )
        out = Path(tmp) / src_48k.name
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"deep-filter failed: {proc.stderr.strip()[-500:]}")
        shutil.move(str(out), dst)
    return dst
