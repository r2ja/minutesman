# LLM stages: fuse both passes into Roman Urdu, then name speakers and split meetings
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

ANALYZE_SYSTEM = """\
You analyse a diarized transcript of ONE phone recording that may contain SEVERAL meetings: the
person recording walked between meeting rooms, so there can be hallway chatter, long silences and
different groups of people. Speaker ids (S1, S2...) are anonymous and consistent across the file.

Task 1: speaker names. Infer each speaker's real name only from explicit evidence:
self-introductions ("main Bilal"), being addressed right before they reply ("Sara, aap batayein?"
then S2 answers), or being thanked or named for something they just said. Being mentioned in the
third person is weak evidence. Never guess from gender, role or language. If evidence is weak or
conflicting, return null with low confidence. Two ids should not get the same name unless the
evidence clearly says so. Judge names PER MEETING: give one entry for every (speaker id, meeting)
pair where that id speaks, using the meeting's position in your meetings list (-1 for lines
outside any meeting). The diarizer can give two similar voices from different rooms the same id,
so the same id may be a different named person in another meeting; name each one separately.

Task 2: meetings. Split the recording into meetings by line numbers. Boundaries show up as
greetings and openings ("Assalam o alaikum", "chalein shuru karte hain"), closings ("thank you
sab ka", "theek hai phir"), long gaps (marked in the transcript), the set of speakers changing,
and the topic changing. Lines between meetings (walking, hallway talk, small talk) belong to no
meeting. Do not split one meeting just because the topic moves on. If the whole file is one
meeting, return one meeting. Title each meeting with a few words about its main topic."""


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
    meeting: int = Field(description="position in the meetings list, -1 outside meetings")
    name: str | None
    confidence: float
    evidence: str = Field(description="quote the lines that support the name")


class Meeting(BaseModel):
    first_line: int
    last_line: int
    title: str = Field(description="a few words, in the language of the meeting")
    boundary_evidence: str = Field(description="why it starts/ends here")


class Analysis(BaseModel):
    speakers: list[SpeakerName]
    meetings: list[Meeting]


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
    # Streamed so long reasoning calls keep the connection busy
    with client.responses.stream(
        model=cfg.llm_model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=schema,
        **kwargs,
    ) as stream:
        for _ in stream:
            pass
        resp = stream.get_final_response()
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


GAP_MARK_SECONDS = 30


# One pass over the whole transcript: speaker names and meeting boundaries
def analyze(client: OpenAI, cfg: Settings, segs: list[Segment], speakers: list[Speaker],
            usage: Usage) -> Analysis:
    lines, prev_end = [], None
    for n, s in enumerate(segs):
        if prev_end is not None and s.start - prev_end >= GAP_MARK_SECONDS:
            lines.append(f"--- {(s.start - prev_end) / 60:.1f} min without speech ---")
        lines.append(f"L{n} [{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.speaker}: {s.text}")
        prev_end = s.end
    ids = ", ".join(sp.id for sp in speakers)
    user = (f"Context: {cfg.context or 'n/a'}\nSpeaker ids: {ids}\nReturn speakers entries per id and meeting. "
            f"Lines are L0..L{len(segs) - 1}.\n\nTranscript:\n" + "\n".join(lines))
    return _parse(client, cfg, ANALYZE_SYSTEM, user, Analysis, usage)
