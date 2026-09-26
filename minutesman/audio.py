"""Audio I/O via a bundled ffmpeg binary (imageio-ffmpeg), so nothing needs to be
installed system-wide on Windows, macOS or Linux."""
from __future__ import annotations

import base64
import io
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# Speech-focused cleanup for phone recordings moved between rooms. Order matters:
# gain first, so far/quiet talkers are lifted before denoising. afftdn works against
# an absolute noise floor, and speech sitting at -45 dBFS would be removed as "noise"
# if it ran first. Denoising is kept mild: ASR models cope with noise much better
# than with denoiser artifacts or near-silent speech.
#   highpass/lowpass  - drop rumble and hiss outside the speech band
#   dynaudnorm (1st)  - per-window gain, up to 30x, lifts quiet talkers without clipping loud ones
#   afftdn            - spectral denoise with noise-floor tracking
#   dynaudnorm (2nd)  - small final level trim
ENHANCE_FILTERS = {
    "off": "anull",
    "light": (
        "highpass=f=70,lowpass=f=7800,dynaudnorm=f=150:g=15:p=0.9:m=30,"
        "afftdn=nf=-25:tn=1,dynaudnorm=f=250:g=15:p=0.9:m=4"
    ),
    "strong": (
        "highpass=f=90,lowpass=f=7500,dynaudnorm=f=150:g=11:p=0.95:m=40,afftdn=nr=20:nf=-25:tn=1,"
        "acompressor=threshold=-24dB:ratio=3:attack=5:release=120,dynaudnorm=f=250:g=15:p=0.9:m=4"
    ),
    # Neural denoisers (see denoise.py), same gain-first order.
    "rnnoise": (
        "highpass=f=70,lowpass=f=7800,dynaudnorm=f=150:g=15:p=0.9:m=30,"
        "arnndn=m=rnnoise-sh.rnnn,dynaudnorm=f=250:g=15:p=0.9:m=4"
    ),
    "deepfilter": None,  # external binary; handled in preprocess()
}
_GAIN = "highpass=f=70,lowpass=f=7800,dynaudnorm=f=150:g=15:p=0.9:m=30"
_TRIM = "dynaudnorm=f=250:g=15:p=0.9:m=4"


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str], input_bytes: bytes | None = None, cwd: Path | None = None) -> bytes:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *args],
        input=input_bytes,
        capture_output=True,
        cwd=cwd,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def duration(path: Path) -> float:
    """Duration in seconds (decodes headers via ffmpeg; no ffprobe needed)."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace"
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        raise RuntimeError(f"Could not read duration of {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def preprocess(src: Path, dst: Path, enhance: str = "light") -> Path:
    """Decode anything ffmpeg understands to 16 kHz mono WAV, with optional cleanup."""
    if enhance not in ENHANCE_FILTERS:
        raise ValueError(f"enhance must be one of {sorted(ENHANCE_FILTERS)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if enhance == "deepfilter":
        from . import denoise

        with tempfile.TemporaryDirectory() as tmp:
            pre, den = Path(tmp) / "pre48.wav", Path(tmp) / "den48.wav"
            # -3 dB headroom: DeepFilterNet clips on full-scale input.
            _run(["-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "48000",
                  "-af", _GAIN + ",volume=-3dB", "-c:a", "pcm_s16le", str(pre)])
            denoise.deepfilter(pre, den)
            _run(["-y", "-i", str(den), "-ac", "1", "-ar", str(SAMPLE_RATE),
                  "-af", _TRIM, "-c:a", "pcm_s16le", str(dst.resolve())])
        return dst
    cwd = None
    if enhance == "rnnoise":
        from . import denoise

        # Run from the model's folder: a Windows path ("C:\...") breaks ffmpeg filter syntax.
        cwd = denoise.rnnoise_model().parent
    _run(["-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
          "-af", ENHANCE_FILTERS[enhance], "-c:a", "pcm_s16le", str(dst.resolve())], cwd=cwd)
    return dst


def load_pcm(path: Path) -> np.ndarray:
    """Whole file as float32 mono at 16 kHz."""
    raw = _run(["-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"])
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def slice_pcm(pcm: np.ndarray, start: float, end: float) -> np.ndarray:
    a = max(0, int(start * SAMPLE_RATE))
    b = min(len(pcm), int(end * SAMPLE_RATE))
    return pcm[a:b]


def pcm_to_wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def pcm_to_mp3_bytes(pcm: np.ndarray, bitrate: str = "64k") -> bytes:
    """Compact upload format: 10 min of 64 kbps mono is ~4.8 MB, well under the 25 MB cap."""
    return _run(["-f", "wav", "-i", "-", "-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", "-"],
                input_bytes=pcm_to_wav_bytes(pcm))


def wav_data_url(pcm: np.ndarray) -> str:
    return "data:audio/wav;base64," + base64.b64encode(pcm_to_wav_bytes(pcm)).decode()


def level_dbfs(pcm: np.ndarray) -> float:
    """RMS level of the active (non-silent) part of a clip, in dBFS."""
    if pcm.size == 0:
        return -120.0
    frame = SAMPLE_RATE // 50  # 20 ms
    n = pcm.size // frame
    if n == 0:
        rms = float(np.sqrt(np.mean(pcm**2)))
    else:
        frames = pcm[: n * frame].reshape(n, frame)
        e = np.sqrt(np.mean(frames**2, axis=1))
        active = e[e >= np.percentile(e, 50)]
        rms = float(np.sqrt(np.mean(active**2))) if active.size else 0.0
    return float(20 * np.log10(max(rms, 1e-6)))


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    # Segments whose midpoint falls in [keep_start, keep_end) belong to this chunk;
    # the rest of the overlap is only used to link speakers with the neighbour.
    keep_start: float
    keep_end: float


def plan_chunks(total: float, chunk_seconds: int, overlap_seconds: int) -> list[Chunk]:
    if chunk_seconds <= overlap_seconds:
        raise ValueError("chunk_seconds must exceed overlap_seconds")
    step = chunk_seconds - overlap_seconds
    chunks: list[Chunk] = []
    start = 0.0
    while True:
        end = min(total, start + chunk_seconds)
        # Avoid a tiny trailing chunk: stretch the last full one instead.
        if total - end < overlap_seconds * 2 and end < total:
            end = total
        chunks.append(Chunk(len(chunks), start, end, 0.0, 0.0))
        if end >= total:
            break
        start += step
    for i, c in enumerate(chunks):
        c.keep_start = 0.0 if i == 0 else (c.start + chunks[i - 1].end) / 2
        c.keep_end = total if i == len(chunks) - 1 else (chunks[i + 1].start + c.end) / 2
    return chunks
