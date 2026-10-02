# Pipeline: enhance, diarize, link speakers, voiceprints, transcribe, fuse, analyze, write (API results cached)
from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI

from .net import make_client

from . import asr, audio, llm, render, voiceprint
from .progress import Counter, fmt, heartbeat, hedged
from .config import LLM_PRICE_PER_1M, TRANSCRIBE_PRICE_PER_MIN, Settings
from .models import Segment, Window
from .speakers import SpeakerRegistry

log = logging.getLogger(__name__)
WINDOW_PAD = 0.3  # seconds of context around each pass-B window


class _Cache:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def get(self, name: str):
        p = self.root / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, name: str, data) -> None:
        tmp = self.root / f"{name}.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.root / f"{name}.json")


def run(src: Path, out_dir: Path, cfg: Settings, voices: dict[str, Path] | None = None,
        fresh: bool = False, client: OpenAI | None = None) -> Path:
    t0 = time.time()
    client = client or make_client()
    work = out_dir / "work"
    if fresh and work.exists():
        shutil.rmtree(work)
    # A different start trim shifts every timestamp, so it gets its own cache
    start = cfg.trim_start or 0.0
    cache = _Cache(work / ("cache" if not start else f"cache_from_{int(start)}s"))

    # 1. Audio
    tag = "" if not (start or cfg.trim_end) else f"_{int(start)}-{int(cfg.trim_end or 0)}"
    enhanced_path = work / f"enhanced_{cfg.enhance}{tag}.wav"
    if not enhanced_path.exists():
        log.info("Preprocessing audio (enhance=%s)", cfg.enhance)
        audio.preprocess(src, enhanced_path, cfg.enhance, start, cfg.trim_end)
    pcm = audio.load_pcm(enhanced_path)
    raw_pcm = audio.load_pcm(src)  # loudness is judged on the untouched recording
    raw_pcm = audio.slice_pcm(raw_pcm, start, cfg.trim_end or len(raw_pcm) / audio.SAMPLE_RATE)
    total = len(pcm) / audio.SAMPLE_RATE
    chunks = audio.plan_chunks(total, cfg.chunk_seconds, cfg.overlap_seconds)
    log.info("Audio: %.1f min, %d chunk(s)", total / 60, len(chunks))

    # 2. Voice samples
    registry = SpeakerRegistry(max_known=cfg.max_known_speakers)
    enrolled_pcm = {}
    for name, path in (voices or {}).items():
        sample = audio.load_pcm(audio.preprocess(path, work / "voices" / f"{name}.wav", cfg.enhance))
        enrolled_pcm[registry.enroll(name, sample)] = sample

    # 3. Pass A, sequential: each chunk is told about the speakers found so far
    per_chunk: list[list[Segment]] = []
    minutes_a = 0.0
    sent_seconds: list[float] = []
    for ch in chunks:
        chunk_pcm = audio.slice_pcm(pcm, ch.start, ch.end)
        names, refs = registry.known_references()
        cached = cache.get(f"passA_{ch.index:03d}")
        if cached and cached.get("span", [ch.start, ch.end]) != [ch.start, ch.end]:
            cached = None  # chunk layout changed since this was cached
        if cached is None:
            log.info("Pass A: chunk %d/%d (%s known speaker refs)", ch.index + 1, len(chunks), len(names))
            t_chunk = time.time()
            mp3 = audio.pcm_to_mp3_bytes(chunk_pcm)
            # Normal chunks take ~4 min; hedge at 1.5x the typical time seen so far
            typical = sorted(sent_seconds)[len(sent_seconds) // 2] if sent_seconds else 240.0
            heard = {"end": ch.start}

            def seen(t, heard=heard, start=ch.start):
                heard["end"] = max(heard["end"], start + t)

            def status(heard=heard, ch=ch):
                return f"heard up to {fmt(heard['end'] - ch.start)} of {fmt(ch.end - ch.start)}"

            with heartbeat(f"pass A chunk {ch.index + 1}/{len(chunks)}", status=status):
                segs = hedged(lambda: asr.diarize_chunk(client, cfg, mp3, ch.index, ch.start, names, refs, seen),
                              max(cfg.hedge_min_seconds, 1.5 * typical), f"Pass A chunk {ch.index + 1}")
            cache.put(f"passA_{ch.index:03d}", {"span": [ch.start, ch.end], "referenced": names,
                                                "segments": [asdict(s) for s in segs]})
            minutes_a += (ch.end - ch.start) / 60  # billed only when actually sent
            sent_seconds.append(time.time() - t_chunk)
            left = len(chunks) - ch.index - 1
            eta = sum(sent_seconds) / len(sent_seconds) * left
            log.info("Pass A: chunk %d done in %s, %d segments; ~%s left for pass A",
                     ch.index + 1, fmt(sent_seconds[-1]), len(segs), fmt(eta))
        else:
            names = cached["referenced"]
            segs = [Segment(**s) for s in cached["segments"]]
            log.info("Pass A: chunk %d/%d from cache", ch.index + 1, len(chunks))
        registry.link_chunk(ch.index, segs, per_chunk[-1] if per_chunk else [], names)
        registry.absorb(segs, chunk_pcm, ch.start)
        per_chunk.append(segs)

    # Keep each segment once: overlapping audio belongs to the chunk whose half it is in.
    segments: list[Segment] = []
    for ch, segs in zip(chunks, per_chunk):
        segments += [s for s in segs if ch.keep_start <= (s.start + s.end) / 2 < ch.keep_end]
    segments.sort(key=lambda s: s.start)
    for s in segments:
        s.level_dbfs = round(audio.level_dbfs(audio.slice_pcm(raw_pcm, s.start, s.end)), 1)
    del raw_pcm  # ~300 MB for 80 minutes; not needed any more

    # 4. Voiceprints
    if cfg.voiceprints and voiceprint.available():
        emb = voiceprint.Embedder(work / "models")
        voiceprint.refine(segments, pcm, emb, {sid: emb.embed(p) for sid, p in enrolled_pcm.items()},
                          registry, cache_path=work / "cache" / "voiceprints.npz")
    elif cfg.voiceprints:
        log.info("Voiceprints skipped (install with: pip install -e \".[voiceprint]\")")

    # 5. Pass B on short windows, in parallel
    windows: list[Window] = []
    for ch in chunks:
        windows += asr.plan_windows([s for s in segments if s.chunk == ch.index], ch.index,
                                    cfg.window_seconds, len(windows))
    prev_text = {w.id: " ".join(s.text_a for s in segments if s.window == w.id - 1)[-300:] for w in windows}

    # Returns billed minutes, 0 when cached
    def pass_b(w: Window) -> float:
        cached = cache.get(f"passB_{w.id:04d}")
        if cached and cached["start"] == w.start and cached["end"] == w.end:
            w.text_b, w.languages = cached["text_b"], cached["languages"]
            return 0.0
        clip = audio.slice_pcm(pcm, w.start - WINDOW_PAD, w.end + WINDOW_PAD)
        asr.transcribe_window(client, cfg, audio.pcm_to_mp3_bytes(clip), w, prev_text[w.id])
        cache.put(f"passB_{w.id:04d}", asdict(w))
        return (w.end - w.start + 2 * WINDOW_PAD) / 60

    def pass_b_counted(w: Window) -> float:
        billed = pass_b(w)
        counter_b.tick()
        return billed

    log.info("Pass B: %d windows", len(windows))
    counter_b = Counter("Pass B", len(windows))
    with ThreadPoolExecutor(cfg.concurrency) as pool:
        minutes_b = sum(pool.map(pass_b_counted, windows))

    # 6. LLM fusion per chunk, in parallel
    usage = llm.Usage()

    def fuse(ch) -> None:
        segs = [s for s in segments if s.chunk == ch.index]
        if not segs:
            return
        key = f"fuse_{ch.index:03d}"
        cached = cache.get(key)
        ids_now = [s.id for s in segs]
        if cached and cached["ids"] == ids_now and cached["speakers"] == [s.speaker for s in segs]:
            fused = {k: llm.FusedSegment(**v) for k, v in cached["fused"].items()}
        else:
            before = [s for s in segments if s.end <= segs[0].start][-6:]
            tail = "\n".join(f"{s.speaker}: {s.text_a}" for s in before)
            with heartbeat(f"LLM fusion chunk {ch.index + 1}", every=60):
                fused = llm.fuse_chunk(client, cfg, segs, [w for w in windows if w.chunk == ch.index], tail, usage)
            cache.put(key, {"ids": ids_now, "speakers": [s.speaker for s in segs],
                            "fused": {k: v.model_dump() for k, v in fused.items()}})
        present = {s.speaker for s in segs}
        for s in segs:
            f = fused.get(s.id)
            if f is None:
                s.text = s.text_a
                continue
            s.text, s.text_confidence = f.text.strip(), round(f.text_confidence, 3)
            s.llm_confidence = round(f.speaker_confidence, 3)
            if f.note:
                s.notes = (s.notes + " " + f.note).strip()
            if (f.suggested_speaker in present and f.suggested_speaker != s.speaker
                    and f.speaker_confidence < 0.35 and (s.acoustic_confidence or 0) < 0.6):
                s.notes = (s.notes + f" LLM moved from {s.speaker};").strip()
                s.speaker = f.suggested_speaker
                s.llm_confidence = 0.6  # its confidence was about the old label
        counter_f.tick()

    log.info("LLM fusion with %s: %d chunks", cfg.llm_model, len(chunks))
    counter_f = Counter("LLM fusion", len(chunks))
    with ThreadPoolExecutor(cfg.concurrency) as pool:
        list(pool.map(fuse, chunks))
    segments = [s for s in segments if s.text]
    absorb_minor_speakers(segments, cfg.min_speaker_seconds)

    # 7. Confidence, names, labels
    for s in segments:
        s.confidence = combine_confidence(s)
    active = {s.speaker for s in segments}
    speakers = [sp for sp in registry.speakers if sp.id in active]
    for sp in speakers:
        sp.talk_seconds = round(sum(s.duration for s in segments if s.speaker == sp.id), 1)

    fingerprint = [[s.id, s.speaker, s.text] for s in segments]
    cached = cache.get("analysis")
    analysis = None
    if cached and cached["segments"] == fingerprint:
        try:
            analysis = llm.Analysis(**cached["analysis"])
        except ValueError:
            log.info("Cached analysis is from an older version; redoing it")
    if analysis is None:
        log.info("Analysis: meetings and names over %d segments", len(segments))
        with heartbeat("meeting/name analysis", every=60):
            analysis = llm.analyze(client, cfg, segments, speakers, usage)
        cache.put("analysis", {"segments": fingerprint, "analysis": analysis.model_dump()})
    off_spans = mark_off_meeting(segments, analysis.off_meeting)
    meetings, index_of = split_meetings(segments, analysis.meetings)
    best = resolve_names(segments, speakers, analysis.speakers, index_of, registry, cfg.name_threshold)
    assign_labels(speakers, best, segments, cfg.name_threshold)
    apply_renames(speakers, cfg.rename)
    for m in meetings:
        # Recount after name resolution, which can split one id into two people
        talk: dict[str, float] = {}
        for seg in segments:
            if seg.meeting == m["index"]:
                talk[seg.speaker] = talk.get(seg.speaker, 0.0) + seg.duration
        m["participants"] = sorted(talk, key=talk.get, reverse=True)

    # 8. Outputs
    cost = {
        "pass_a_minutes": round(minutes_a, 2),
        "pass_b_minutes": round(minutes_b, 2),
        "llm_tokens": {"input": usage.input, "cached": usage.cached, "output": usage.output},
    }
    cost["usd"] = round(
        minutes_a * TRANSCRIBE_PRICE_PER_MIN.get(cfg.diarize_model, 0.006)
        + minutes_b * TRANSCRIBE_PRICE_PER_MIN.get(cfg.transcribe_model, 0.006)
        + llm_cost(cfg.llm_model, usage.input, usage.cached, usage.output), 4)
    cost["note"] = "Counts only API calls made in this run; stages served from cache cost nothing."
    meta = {
        "source": str(src), "duration_seconds": round(total, 1), "settings": asdict(cfg),
        "voiceprints": cfg.voiceprints and voiceprint.available(), "cost": cost, "meetings": meetings,
        "off_meeting": off_spans, "trim": [start, cfg.trim_end],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if start:
        shift_timeline(segments, meetings, off_spans, start)
    render.write_all(out_dir, segments, speakers, meta)
    log.info("Done in %.0fs, est. cost $%.3f -> %s", meta["elapsed_seconds"], cost["usd"], out_dir)
    return out_dir


def llm_cost(model: str, inp: int, cached: int, out: int) -> float:
    p_in, p_cached, p_out = LLM_PRICE_PER_1M.get(model, (2.0, 0.2, 10.0))
    return ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6


# Blend voiceprint 0.5, linking 0.3, LLM 0.2 (over sources present), then discount short or quiet segments
def combine_confidence(s: Segment) -> float:
    parts = [(s.link_confidence, 0.3)]
    if s.acoustic_confidence is not None:
        parts.append((s.acoustic_confidence, 0.5))
    if s.llm_confidence is not None:
        parts.append((s.llm_confidence, 0.2))
    c = sum(v * w for v, w in parts) / sum(w for _, w in parts)
    if s.duration < 1.0:
        c *= 0.8
    if s.level_dbfs < -40:
        c *= 0.85
    return round(max(0.0, min(1.0, c)), 3)


# Clean the LLM's line ranges into non-overlapping meetings; lines outside any stay -1
def split_meetings(segments: list[Segment], proposed) -> list[dict]:
    n = len(segments)
    spans = sorted((max(0, m.first_line), min(n - 1, m.last_line), i, m) for i, m in enumerate(proposed)
                   if m.first_line <= m.last_line and m.first_line < n)
    kept, last_end = [], -1
    for a, b, i, m in spans:
        a = max(a, last_end + 1)
        if a <= b:
            kept.append((a, b, i, m))
            last_end = b
    if not kept and n:
        kept = [(0, n - 1, -2, None)]
    meetings, index_of = [], {-1: -1}
    for k, (a, b, i, m) in enumerate(kept):
        index_of[i] = k
        members = [s for s in segments[a:b + 1] if not s.off_reason] or segments[a:b + 1]
        for s in members:
            s.meeting = k
        talk: dict[str, float] = {}
        for s in members:
            talk[s.speaker] = talk.get(s.speaker, 0.0) + s.duration
        meetings.append({
            "index": k, "title": m.title if m else "Meeting",
            "start": members[0].start, "end": members[-1].end,
            "participants": sorted(talk, key=talk.get, reverse=True),
            "boundary_evidence": m.boundary_evidence if m else "",
        })
    return meetings, index_of


# Tag lines the analysis put outside the meetings; returns [{start, end, reason}] for the report
def mark_off_meeting(segments: list[Segment], spans) -> list[dict]:
    out = []
    n = len(segments)
    for sp in sorted(spans, key=lambda x: x.first_line):
        a, b = max(0, sp.first_line), min(n - 1, sp.last_line)
        if a > b or a >= n:
            continue
        for s in segments[a:b + 1]:
            s.off_reason = sp.reason.strip() or "outside the meetings"
        out.append({"start": segments[a].start, "end": segments[b].end, "reason": segments[a].off_reason,
                    "lines": b - a + 1})
    return out


# Move everything back onto the original recording's clock after a start trim
def shift_timeline(segments: list[Segment], meetings: list[dict], off_spans: list[dict], offset: float) -> None:
    for s in segments:
        s.start, s.end = round(s.start + offset, 2), round(s.end + offset, 2)
    for m in [*meetings, *off_spans]:
        m["start"], m["end"] = round(m["start"] + offset, 2), round(m["end"] + offset, 2)


# Split an id that is confidently a different named person in different meetings; return best name per id
def resolve_names(segments, speakers, entries, index_of: dict, registry, threshold: float) -> dict:
    by_id: dict[str, list] = {}
    for e in entries:
        by_id.setdefault(e.speaker, []).append((index_of.get(e.meeting, -1), e))
    for sp in list(speakers):
        confident: dict[str, list] = {}
        for m, e in by_id.get(sp.id, []):
            if e.name and e.confidence >= threshold and m >= 0:
                confident.setdefault(e.name.strip().lower(), []).append((m, e))
        if sp.enrolled or len(confident) < 2:
            continue
        talk = {k: sum(s.duration for s in segments if s.speaker == sp.id and s.meeting in {m for m, _ in v})
                for k, v in confident.items()}
        keep = max(talk, key=talk.get)
        for name, hits in confident.items():
            if name == keep:
                continue
            new = registry._new().speaker
            moved_meetings = {m for m, _ in hits}
            for s in segments:
                if s.speaker == sp.id and s.meeting in moved_meetings:
                    s.speaker = new.id
                    s.notes = (s.notes + f" split from {sp.id}: named differently in this meeting;").strip()
            new.talk_seconds = round(sum(s.duration for s in segments if s.speaker == new.id), 1)
            sp.talk_seconds = round(sp.talk_seconds - new.talk_seconds, 1)
            speakers.append(new)
            by_id[new.id] = [(m, e) for m, e in hits]
            by_id[sp.id] = [(m, e) for m, e in by_id[sp.id] if m not in moved_meetings]
            log.info("Names: %s is two people across meetings; split %s into %s", sp.id, hits[0][1].name, new.id)
    best = {}
    for sid, hits in by_id.items():
        named = [e for _, e in hits if e.name]
        if named:
            best[sid] = max(named, key=lambda e: e.confidence)
    return best


# Speakers with a few seconds in total are fragments of real speakers: fold them into the nearest one in time
def absorb_minor_speakers(segments: list[Segment], min_seconds: float) -> None:
    talk: dict[str, float] = {}
    for s in segments:
        talk[s.speaker] = talk.get(s.speaker, 0.0) + s.duration
    # Short recordings have short real speakers, so the bar is at most 2% of all speech
    min_seconds = min(min_seconds, 0.02 * sum(talk.values()))
    minor = {k for k, v in talk.items() if v < min_seconds and not k.startswith("E")}
    major = [s for s in segments if s.speaker not in minor]
    if not minor or not major:
        return
    for s in segments:
        if s.speaker not in minor:
            continue

        def gap(o, s=s):
            return max(o.start - s.end, s.start - o.end, 0.0)

        same = [o for o in major if o.chunk == s.chunk and o.local_speaker == s.local_speaker and gap(o) <= 60]
        near = min(same or major, key=gap)
        s.notes = (s.notes + f" fragment speaker {s.speaker} folded into {near.speaker};").strip()
        s.speaker = near.speaker
        s.confidence = round(min(s.confidence, 0.55), 3)
    log.info("Folded %d fragment speakers (< %.0fs of speech each) into their nearest real speaker",
             len(minor), min_seconds)


# User overrides like {"S2": "Raja"}: the name wins over whatever the analysis inferred
def apply_renames(speakers, rename: dict) -> None:
    for sp in speakers:
        if sp.id in rename:
            sp.label, sp.name_guess, sp.name_confidence = rename[sp.id], rename[sp.id], 1.0
            sp.name_evidence = "set by user"


# Real names only above threshold and never twice; everyone else is Guest N by first appearance
def assign_labels(speakers, names: dict, segments: list[Segment], threshold: float) -> None:
    for sp in speakers:
        n = names.get(sp.id)
        if not sp.enrolled and n is not None:
            sp.name_guess, sp.name_confidence, sp.name_evidence = n.name, round(n.confidence, 3), n.evidence
    taken: dict[str, str] = {}
    for sp in sorted(speakers, key=lambda x: (not x.enrolled, -x.name_confidence)):
        name = (sp.name_guess or "").strip()
        if name and (sp.enrolled or sp.name_confidence >= threshold) and name.lower() not in taken:
            sp.label = name
            taken[name.lower()] = sp.id
    first_seen = {}
    for s in segments:
        first_seen.setdefault(s.speaker, s.start)
    guest = 0
    for sp in sorted(speakers, key=lambda x: first_seen.get(x.id, 1e12)):
        if not sp.label:
            guest += 1
            sp.label = f"Guest {guest}"
