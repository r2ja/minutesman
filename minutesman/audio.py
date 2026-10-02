# Audio I/O through the bundled ffmpeg binary, no system install needed
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

# Cleanup chains: gain first (lifts far/quiet talkers), then mild denoise, else quiet speech is removed as noise
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


# Duration in seconds, read from ffmpeg's header output
def duration(path: Path) -> float:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace"
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not m:
        raise RuntimeError(f"Could not read duration of {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


# Decode to 16 kHz mono WAV with the chosen cleanup
def preprocess(src: Path, dst: Path, enhance: str = "light", start: float = 0.0, end: float | None = None) -> Path:
    trim = (["-ss", f"{start:.2f}"] if start else []) + (["-to", f"{end:.2f}"] if end else [])
    if enhance not in ENHANCE_FILTERS:
        raise ValueError(f"enhance must be one of {sorted(ENHANCE_FILTERS)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if enhance == "deepfilter":
        from . import denoise

        with tempfile.TemporaryDirectory() as tmp:
            pre, den = Path(tmp) / "pre48.wav", Path(tmp) / "den48.wav"
            # -3 dB headroom: DeepFilterNet clips on full-scale input.
            _run(["-y", *trim, "-i", str(src), "-vn", "-ac", "1", "-ar", "48000",
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
    _run(["-y", *trim, "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
          "-af", ENHANCE_FILTERS[enhance], "-c:a", "pcm_s16le", str(dst.resolve())], cwd=cwd)
    return dst


# Small speech-quality Opus copy for uploading (80 min at 32 kbps is ~19 MB)
def shrink(src: Path, dst: Path, bitrate: str = "32k") -> Path:
    _run(["-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "libopus",
          "-b:a", bitrate, "-application", "voip", str(dst)])
    return dst


# Whole file as float32 mono at 16 kHz
# Cut [start, end) seconds to a 16 kHz mono WAV, e.g. a voice sample for --voice
def clip(src: Path, dst: Path, start: float, end: float) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["-y", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{end - start:.2f}", "-vn", "-ac", "1",
          "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)])
    return dst


def load_pcm(path: Path) -> np.ndarray:
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


# Upload format: 10 min at 64 kbps mono is ~4.8 MB, under the 25 MB cap
def pcm_to_mp3_bytes(pcm: np.ndarray, bitrate: str = "64k") -> bytes:
    return _run(["-f", "wav", "-i", "-", "-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", "-"],
                input_bytes=pcm_to_wav_bytes(pcm))


def wav_data_url(pcm: np.ndarray) -> str:
    return "data:audio/wav;base64," + base64.b64encode(pcm_to_wav_bytes(pcm)).decode()


# RMS level of the louder half of 20 ms frames, in dBFS
def level_dbfs(pcm: np.ndarray) -> float:
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
    # A segment belongs to the chunk whose [keep_start, keep_end) holds its midpoint
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
