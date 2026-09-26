# Settings; any field can be overridden with MINUTESMAN_<NAME> or a CLI flag
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

# USD list prices (developers.openai.com, 2026-09): per minute for ASR, per 1M tokens (in, cached, out) for LLMs
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
    pass_a_timeout: int = 600  # seconds per diarization request before the client retries it
    hedge_min_seconds: int = 300  # never send a second copy of a chunk before this

    # Minimum naming confidence to publish a real name instead of "Guest N"
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
