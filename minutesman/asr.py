"""The two speech-to-text passes.

Pass A (`diarize_chunk`): gpt-4o-transcribe-diarize on ~10 min chunks. Gives who
spoke when, plus a first text hypothesis.

Pass B (`transcribe_window`): gpt-transcribe on ~90 s windows cut on pass-A turn
boundaries. It is the stronger text model (code-switching, language hints,
keywords) but returns no timestamps, so short windows keep it aligned with pass A.
"""
from __future__ import annotations

import io
import logging
import re

from openai import BadRequestError, NotFoundError, OpenAI

from .config import Settings
from .models import Segment, Window

log = logging.getLogger(__name__)


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _prompt_safe(text: str, limit: int = 900) -> str:
    """gpt-transcribe rejects '<', '>' and line breaks in prompt/keywords."""
    text = re.sub(r"[<>\r\n]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[-limit:]


def diarize_chunk(client: OpenAI, cfg: Settings, mp3: bytes, chunk_index: int, offset: float,
                  known_names: list[str], known_refs: list[str]) -> list[Segment]:
    kwargs = {}
    if known_names:
        kwargs["known_speaker_names"] = known_names
        kwargs["known_speaker_references"] = known_refs
    resp = client.audio.transcriptions.create(
        model=cfg.diarize_model,
        file=(f"chunk{chunk_index:03d}.mp3", io.BytesIO(mp3), "audio/mpeg"),
        response_format="diarized_json",
        chunking_strategy="auto",
        **kwargs,
    )
    segs = []
    for i, s in enumerate(_field(resp, "segments", []) or []):
        text = (_field(s, "text", "") or "").strip()
        if not text:
            continue
        segs.append(Segment(
            id=f"c{chunk_index:02d}s{i:04d}",
            chunk=chunk_index,
            start=round(offset + float(_field(s, "start", 0.0)), 2),
            end=round(offset + float(_field(s, "end", 0.0)), 2),
            local_speaker=str(_field(s, "speaker", "?")),
            text_a=text,
        ))
    return segs


def transcribe_window(client: OpenAI, cfg: Settings, mp3: bytes, window: Window,
                      previous_text: str = "") -> Window:
    prompt = _prompt_safe(
        f"{cfg.context} Office meeting recorded on a phone in Pakistan. Speakers mix Urdu "
        f"and English mid-sentence. Transcribe exactly what is said, in the language spoken. "
        f"Previous: {previous_text}"
    )
    extra = {"languages": cfg.languages}
    if cfg.keywords:
        extra["keywords"] = [_prompt_safe(k, 100) for k in cfg.keywords][:100]
    model = cfg.transcribe_model
    try:
        resp = client.audio.transcriptions.create(
            model=model, file=(f"w{window.id:04d}.mp3", io.BytesIO(mp3), "audio/mpeg"),
            prompt=prompt, response_format="json", **extra,
        )
    except (BadRequestError, NotFoundError) as exc:
        # Older accounts or regions may lack gpt-transcribe; its successor-era params
        # (languages/keywords) are also unknown to older models.
        log.warning("%s failed (%s); falling back to %s", model, exc, cfg.fallback_transcribe_model)
        resp = client.audio.transcriptions.create(
            model=cfg.fallback_transcribe_model,
            file=(f"w{window.id:04d}.mp3", io.BytesIO(mp3), "audio/mpeg"),
            prompt=prompt, response_format="json",
        )
    window.text_b = (_field(resp, "text", "") or "").strip()
    window.languages = [str(_field(lang, "code", lang)) for lang in (_field(resp, "languages", []) or [])]
    return window


def plan_windows(segs: list[Segment], chunk_index: int, target: float, first_id: int) -> list[Window]:
    """Group consecutive segments into windows of about `target` seconds, cutting only
    between segments so each window's text maps cleanly to its segments."""
    windows: list[Window] = []
    cur: list[Segment] = []
    for s in segs:
        if cur and s.end - cur[0].start > target:
            windows.append(Window(first_id + len(windows), chunk_index, cur[0].start, cur[-1].end))
            cur = []
        cur.append(s)
        s.window = first_id + len(windows)
    if cur:
        windows.append(Window(first_id + len(windows), chunk_index, cur[0].start, cur[-1].end))
    return windows
