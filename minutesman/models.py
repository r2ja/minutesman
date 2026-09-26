from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Segment:
    id: str
    chunk: int
    start: float  # absolute seconds in the recording
    end: float
    local_speaker: str  # label returned by the diarizer for this chunk
    text_a: str  # pass A (diarizer) hypothesis
    speaker: str = ""  # global speaker id, e.g. "S1"
    window: int = -1  # pass-B window this segment belongs to
    level_dbfs: float = 0.0  # loudness in the original (un-enhanced) recording
    link_confidence: float = 0.0  # how sure the chunk->global speaker link is
    acoustic_confidence: float | None = None  # voiceprint similarity, if available
    llm_confidence: float | None = None
    confidence: float = 0.0  # final diarization confidence
    text: str = ""  # fused Roman Urdu / English text
    text_confidence: float | None = None
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Window:
    id: int
    chunk: int
    start: float
    end: float
    text_b: str = ""  # pass B hypothesis
    languages: list[str] = field(default_factory=list)


@dataclass
class Speaker:
    id: str  # "S1"
    label: str = ""  # published label: a real name or "Guest N"
    name_guess: str | None = None
    name_confidence: float = 0.0
    name_evidence: str = ""
    enrolled: bool = False  # matched a voice sample supplied by the user
    talk_seconds: float = 0.0


def to_dict(obj):
    return asdict(obj)
