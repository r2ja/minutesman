"""LLM stages: fuse the two ASR passes into Roman Urdu, then name the speakers."""
from __future__ import annotations

import json
import logging

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .models import Segment, Speaker, Window

log = logging.getLogger(__name__)

ROMAN_URDU_RULES = """\
Output language rules (Roman Urdu + English):
- Write Urdu in Roman Urdu as Pakistanis type it: "main", "hum", "aap", "hai", "hain", "nahi",
  "kya", "kyun", "kab", "mein", "ko", "se", "ka/ki/ke", "abhi", "theek hai", "bilkul",
  "kar rahe hain", "chahiye", "InshaAllah", "Assalam o Alaikum".
- Keep English words and sentences in normal English spelling. Never translate between the
  languages; keep the speaker's own code-switching exactly as spoken.
- Never output Urdu/Arabic script or Devanagari. If a hypothesis is in Urdu or Hindi script,
  transliterate it into Roman Urdu (Hindi-script output is the ASR mis-hearing Urdu).
- Numbers, names, product and company names: keep as spoken ("twenty percent", "Q3").
- Keep meaning-bearing fillers ("haan", "achha", "okay") but drop pure noise ("uh", "umm")."""

FUSE_SYSTEM = f"""\
You clean up meeting transcripts from Pakistani offices. You receive, for one stretch of a
recording, two independent speech-recognition hypotheses:
- Pass A: diarized segments (id, time, speaker id, text). Its segmentation and timing are fixed.
- Pass B: a usually more accurate text for each ~90 s window, without speaker labels.
Audio was recorded on a phone moving between meeting rooms: some parts are far away, quiet or
distorted, so either pass can contain mishearings, hallucinated phrases or dropped words.

For every pass-A segment, write the best text for exactly that segment by aligning it with the
matching part of its window's pass-B text: prefer pass B wording, use pass A where B is missing
or garbled, and never invent content that neither pass supports. Every pass-B word should end up
in some segment of its window. If a segment is pure noise or a hallucination, return empty text.

Also judge the speaker label from the conversation itself (questions answered by someone else,
people addressing each other by name, "I'll do it" commitments, a turn that obviously contains
two people). Give speaker_confidence in [0,1] that the pass-A speaker id is right, and only if
you are confident it is wrong, name the better id from the ids present in this stretch.

{ROMAN_URDU_RULES}"""

NAME_SYSTEM = """\
You identify meeting participants. Given a diarized transcript with anonymous speaker ids, infer
each speaker's real name only from explicit evidence: self-introductions ("main Bilal"), being
addressed right before they reply ("Sara, aap batayein?" then S2 answers), or being thanked/
named for something they said. Being mentioned in the third person is weak evidence. Never guess
from gender, role or language. If evidence is weak or conflicting, return null with low
confidence. Two ids should not get the same name unless evidence clearly says so."""


class FusedSegment(BaseModel):
    id: str
    text: str
    text_confidence: float = Field(description="0-1: how sure the text is right")
    speaker_confidence: float = Field(description="0-1: pass-A speaker id is correct")
    suggested_speaker: str | None = Field(description="better speaker id, or null")
    note: str = Field(description="short reason when text or speaker is doubtful, else empty")


class FusedChunk(BaseModel):
    segments: list[FusedSegment]


class SpeakerName(BaseModel):
    speaker: str
    name: str | None
    confidence: float
    evidence: str = Field(description="quote the lines that support the name")


class SpeakerNames(BaseModel):
    speakers: list[SpeakerName]


class Usage:
    def __init__(self):
        self.input = self.cached = self.output = 0

    def add(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if not u:
            return
        self.input += getattr(u, "input_tokens", 0) or 0
        self.output += getattr(u, "output_tokens", 0) or 0
        details = getattr(u, "input_tokens_details", None)
        self.cached += getattr(details, "cached_tokens", 0) or 0


def _parse(client: OpenAI, cfg: Settings, system: str, user: str, schema, usage: Usage):
    kwargs = {}
    if cfg.reasoning_effort and cfg.reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": cfg.reasoning_effort}
    resp = client.responses.parse(
        model=cfg.llm_model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=schema,
        **kwargs,
    )
    usage.add(resp)
    if resp.output_parsed is None:
        raise RuntimeError(f"{cfg.llm_model} returned no parsable output")
    return resp.output_parsed


def fuse_chunk(client: OpenAI, cfg: Settings, segs: list[Segment], windows: list[Window],
               previous_tail: str, usage: Usage) -> dict[str, FusedSegment]:
    payload = {
        "meeting_context": cfg.context or None,
        "keywords": cfg.keywords or None,
        "previous_lines_for_context": previous_tail or None,
        "windows": [
            {
                "window": w.id,
                "pass_b_text": w.text_b,
                "pass_b_languages": w.languages,
                "pass_a_segments": [
                    {"id": s.id, "t": f"{s.start:.1f}-{s.end:.1f}", "speaker": s.speaker,
                     "quiet_audio": s.level_dbfs < -38, "text": s.text_a}
                    for s in segs if s.window == w.id
                ],
            }
            for w in windows
        ],
    }
    out = _parse(client, cfg, FUSE_SYSTEM, json.dumps(payload, ensure_ascii=False), FusedChunk, usage)
    result = {f.id: f for f in out.segments}
    missing = [s.id for s in segs if s.id not in result]
    if missing:
        log.warning("LLM skipped %d segment(s); keeping pass-A text for them", len(missing))
    return result


def name_speakers(client: OpenAI, cfg: Settings, segs: list[Segment], speakers: list[Speaker],
                  usage: Usage) -> dict[str, SpeakerName]:
    lines = "\n".join(f"[{s.start / 60:05.1f}m] {s.speaker}: {s.text}" for s in segs if s.text)
    ids = ", ".join(sp.id for sp in speakers)
    user = (f"Meeting context: {cfg.context or 'n/a'}\nSpeaker ids: {ids}\n"
            f"Return one entry per id.\n\nTranscript:\n{lines}")
    out = _parse(client, cfg, NAME_SYSTEM, user, SpeakerNames, usage)
    return {n.speaker: n for n in out.speakers}
