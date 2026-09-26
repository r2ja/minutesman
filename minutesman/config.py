"""Runtime settings. Every field can be overridden by an env var (MINUTESMAN_<NAME>)
or a CLI flag, so the same code runs unchanged on a laptop and in a container."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

# USD list prices, fetched from developers.openai.com/api/docs/pricing (2026-09).
# Per-minute for transcription, per 1M tokens (input, cached input, output) for LLMs.
TRANSCRIBE_PRICE_PER_MIN = {
    "gpt-transcribe": 0.0045,
    "gpt-4o-transcribe-diarize": 0.006,
    "gpt-4o-transcribe": 0.006,
    "gpt-4o-mini-transcribe": 0.003,
    "whisper-1": 0.006,
}
LLM_PRICE_PER_1M = {
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.6-sol": (4.00, 0.40, 20.00),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5-mini": (0.25, 0.025, 2.00),
}


def _env(name: str, default):
    raw = os.environ.get(f"MINUTESMAN_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


@dataclass
class Settings:
    # Pass A: diarization + first transcript hypothesis.
    diarize_model: str = "gpt-4o-transcribe-diarize"
    # Pass B: best plain-text ASR, run on short windows aligned to pass A turns.
    transcribe_model: str = "gpt-transcribe"
    fallback_transcribe_model: str = "gpt-4o-transcribe"
    # Fusion / Roman Urdu / speaker-naming LLM.
    llm_model: str = "gpt-6-sol"
    reasoning_effort: str = "medium"

    languages: list = field(default_factory=lambda: ["ur", "en"])
    keywords: list = field(default_factory=list)
    context: str = ""  # free-text description of the meeting, fed to ASR and LLM

    enhance: str = "light"  # off | light | strong
    chunk_seconds: int = 600
    overlap_seconds: int = 30
    window_seconds: int = 90  # pass-B window length
    max_known_speakers: int = 4  # API limit for known_speaker_references
    concurrency: int = 4

    # A speaker keeps a real name only if the naming confidence clears this bar;
    # otherwise it is published as "Guest N".
    name_threshold: float = 0.75
    voiceprints: bool = True  # use local speaker embeddings if the extra is installed

    def __post_init__(self):
        for f in fields(self):
            setattr(self, f.name, _env(f.name, getattr(self, f.name)))

    def update(self, **overrides) -> "Settings":
        for k, v in overrides.items():
            if v is not None:
                setattr(self, k, v)
        return self
