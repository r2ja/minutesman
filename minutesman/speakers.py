"""Keeps speaker identities consistent across chunks.

The diarizer labels speakers per request (A, B, C...), so an 80-minute file split
into ten-minute chunks needs its labels stitched together. Three signals are used,
strongest first:

1. Known-speaker references: short clips of speakers already seen are sent with each
   new chunk (API limit: 4), and the diarizer answers with our global ids directly.
2. Overlap voting: consecutive chunks share `overlap_seconds` of audio; a new label
   that talks over the same seconds as a known speaker in the previous chunk is them.
3. Voiceprints (optional, local): see voiceprint.py; refines links after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE, level_dbfs, slice_pcm, wav_data_url
from .models import Segment, Speaker

REF_MIN, REF_MAX = 2.5, 10.0  # API accepts 2-10 s reference clips


@dataclass
class _Entry:
    speaker: Speaker
    ref_clip: np.ndarray | None = None
    ref_score: float = -1e9
    last_chunk: int = -1


@dataclass
class SpeakerRegistry:
    max_known: int = 4
    entries: dict[str, _Entry] = field(default_factory=dict)

    # -- registration -----------------------------------------------------------------
    def _new(self, name: str | None = None, clip: np.ndarray | None = None) -> _Entry:
        sid = f"S{len(self.entries) + 1}"
        spk = Speaker(id=sid)
        if name:
            spk.name_guess, spk.name_confidence, spk.enrolled = name, 1.0, True
            spk.name_evidence = "matched user-supplied voice sample"
        e = _Entry(spk, ref_clip=clip, ref_score=1e9 if clip is not None else -1e9)
        self.entries[sid] = e
        return e

    def enroll(self, name: str, pcm: np.ndarray) -> str:
        """Register a user-supplied voice sample (only the first 10 s are sent)."""
        return self._new(name, pcm[: int(REF_MAX * SAMPLE_RATE)]).speaker.id

    @property
    def speakers(self) -> list[Speaker]:
        return [e.speaker for e in self.entries.values()]

    # -- references for the next diarization request ---------------------------------
    def known_references(self) -> tuple[list[str], list[str]]:
        """Up to `max_known` (ids, data URLs), preferring enrolled, recent, talkative."""
        cands = [e for e in self.entries.values() if e.ref_clip is not None]
        cands.sort(key=lambda e: (e.speaker.enrolled, e.last_chunk, e.speaker.talk_seconds), reverse=True)
        chosen = cands[: self.max_known]
        return [e.speaker.id for e in chosen], [wav_data_url(e.ref_clip) for e in chosen]

    # -- linking ----------------------------------------------------------------------
    def link_chunk(self, chunk_index: int, segs: list[Segment], prev: list[Segment],
                   referenced: list[str]) -> None:
        """Assign global ids to `segs` (one chunk, un-trimmed) in place."""
        labels = sorted({s.local_speaker for s in segs})
        crowded = len(self.entries) > len(referenced)  # someone known wasn't referenced
        mapping: dict[str, tuple[str, float]] = {}
        for label in labels:
            if label in self.entries and label in referenced:
                mapping[label] = (label, 0.85)
                continue
            vote = self._overlap_vote([s for s in segs if s.local_speaker == label], prev)
            if vote:
                mapping[label] = vote
                continue
            mapping[label] = (self._new().speaker.id, 0.55 if crowded else 0.7)

        for s in segs:
            s.speaker, s.link_confidence = mapping[s.local_speaker]
        for sid in {m[0] for m in mapping.values()}:
            self.entries[sid].last_chunk = chunk_index

    @staticmethod
    def _overlap_vote(mine: list[Segment], prev: list[Segment]) -> tuple[str, float] | None:
        votes: dict[str, float] = {}
        for s in mine:
            for p in prev:
                ov = min(s.end, p.end) - max(s.start, p.start)
                if ov > 0 and p.speaker:
                    votes[p.speaker] = votes.get(p.speaker, 0.0) + ov
        total = sum(votes.values())
        if total < 1.5:
            return None
        best, secs = max(votes.items(), key=lambda kv: kv[1])
        share = secs / total
        if share < 0.6:
            return None
        return best, round(0.55 + 0.35 * share, 3)

    # -- bookkeeping ------------------------------------------------------------------
    def absorb(self, segs: list[Segment], pcm: np.ndarray, pcm_offset: float = 0.0) -> None:
        """Update talk time and keep the best reference clip per speaker.

        `pcm` is the enhanced audio of the chunk; `pcm_offset` its absolute start time.
        """
        for s in segs:
            e = self.entries[s.speaker]
            e.speaker.talk_seconds += s.duration
            if e.speaker.enrolled or s.duration < REF_MIN:
                continue
            clip = slice_pcm(pcm, s.start - pcm_offset, min(s.end, s.start + REF_MAX) - pcm_offset)
            # Prefer longer, louder turns: they make the most reliable references.
            score = min(s.duration, REF_MAX) + level_dbfs(clip) / 10
            if score > e.ref_score and clip.size >= REF_MIN * SAMPLE_RATE:
                e.ref_clip, e.ref_score = clip, score

    def merge(self, keep: str, drop: str, segs: list[Segment]) -> None:
        for s in segs:
            if s.speaker == drop:
                s.speaker = keep
        k, d = self.entries[keep], self.entries.pop(drop)
        k.speaker.talk_seconds += d.speaker.talk_seconds
        if d.speaker.enrolled and not k.speaker.enrolled:
            k.speaker.name_guess, k.speaker.name_confidence = d.speaker.name_guess, d.speaker.name_confidence
            k.speaker.enrolled, k.speaker.name_evidence = True, d.speaker.name_evidence
