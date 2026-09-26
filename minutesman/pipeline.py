"""End-to-end run: audio -> enhanced chunks -> pass A (diarize) -> speaker linking ->
voiceprints (optional) -> pass B (windows) -> LLM fusion -> speaker naming -> outputs.

Every API result is cached under <out>/work/, so rerunning after a crash or tweaking
later stages never pays for finished transcription again (use --fresh to redo all).
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI

from . import asr, audio, llm, render, voiceprint
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
    client = client or OpenAI(max_retries=5, timeout=900)
    work = out_dir / "work"
    if fresh and work.exists():
        shutil.rmtree(work)
    cache = _Cache(work / "cache")

    # 1. Audio ------------------------------------------------------------------------
    enhanced_path = work / f"enhanced_{cfg.enhance}.wav"
    if not enhanced_path.exists():
        log.info("Preprocessing audio (enhance=%s)", cfg.enhance)
        audio.preprocess(src, enhanced_path, cfg.enhance)
    pcm = audio.load_pcm(enhanced_path)
    raw_pcm = audio.load_pcm(src)  # loudness is judged on the untouched recording
    total = len(pcm) / audio.SAMPLE_RATE
    chunks = audio.plan_chunks(total, cfg.chunk_seconds, cfg.overlap_seconds)
    log.info("Audio: %.1f min, %d chunk(s)", total / 60, len(chunks))

    # 2. Voice samples ----------------------------------------------------------------
    registry = SpeakerRegistry(max_known=cfg.max_known_speakers)
    enrolled_pcm = {}
    for name, path in (voices or {}).items():
        sample = audio.load_pcm(audio.preprocess(path, work / "voices" / f"{name}.wav", cfg.enhance))
        enrolled_pcm[registry.enroll(name, sample)] = sample

    # 3. Pass A, sequential: each chunk is told about the speakers found so far --------
    per_chunk: list[list[Segment]] = []
    minutes_a = 0.0
    for ch in chunks:
        chunk_pcm = audio.slice_pcm(pcm, ch.start, ch.end)
        names, refs = registry.known_references()
        cached = cache.get(f"passA_{ch.index:03d}")
        if cached is None:
            log.info("Pass A: chunk %d/%d (%s known speaker refs)", ch.index + 1, len(chunks), len(names))
            segs = asr.diarize_chunk(client, cfg, audio.pcm_to_mp3_bytes(chunk_pcm), ch.index,
                                     ch.start, names, refs)
            cache.put(f"passA_{ch.index:03d}", {"referenced": names, "segments": [asdict(s) for s in segs]})
            minutes_a += (ch.end - ch.start) / 60  # billed only when actually sent
        else:
            names = cached["referenced"]
            segs = [Segment(**s) for s in cached["segments"]]
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

    # 4. Voiceprints --------------------------------------------------------------------
    if cfg.voiceprints and voiceprint.available():
        log.info("Voiceprints: embedding %d segments locally", len(segments))
        emb = voiceprint.Embedder(work / "models")
        voiceprint.refine(segments, pcm, emb, {sid: emb.embed(p) for sid, p in enrolled_pcm.items()},
                          registry)
    elif cfg.voiceprints:
        log.info("Voiceprints skipped (install with: pip install -e \".[voiceprint]\")")

    # 5. Pass B on short windows, in parallel --------------------------------------------
    windows: list[Window] = []
    for ch in chunks:
        windows += asr.plan_windows([s for s in segments if s.chunk == ch.index], ch.index,
                                    cfg.window_seconds, len(windows))
    prev_text = {w.id: " ".join(s.text_a for s in segments if s.window == w.id - 1)[-300:] for w in windows}

    def pass_b(w: Window) -> float:
        """Returns the billed minutes (0 when served from cache)."""
        cached = cache.get(f"passB_{w.id:04d}")
        if cached and cached["start"] == w.start and cached["end"] == w.end:
            w.text_b, w.languages = cached["text_b"], cached["languages"]
            return 0.0
        clip = audio.slice_pcm(pcm, w.start - WINDOW_PAD, w.end + WINDOW_PAD)
        asr.transcribe_window(client, cfg, audio.pcm_to_mp3_bytes(clip), w, prev_text[w.id])
        cache.put(f"passB_{w.id:04d}", asdict(w))
        return (w.end - w.start + 2 * WINDOW_PAD) / 60

    log.info("Pass B: %d windows", len(windows))
    with ThreadPoolExecutor(cfg.concurrency) as pool:
        minutes_b = sum(pool.map(pass_b, windows))

    # 6. LLM fusion per chunk, in parallel ------------------------------------------------
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

    log.info("LLM fusion with %s", cfg.llm_model)
    with ThreadPoolExecutor(cfg.concurrency) as pool:
        list(pool.map(fuse, chunks))
    segments = [s for s in segments if s.text]

    # 7. Confidence, names, labels --------------------------------------------------------
    for s in segments:
        s.confidence = combine_confidence(s)
    active = {s.speaker for s in segments}
    speakers = [sp for sp in registry.speakers if sp.id in active]
    for sp in speakers:
        sp.talk_seconds = round(sum(s.duration for s in segments if s.speaker == sp.id), 1)

    cached = cache.get("names")
    if cached and set(cached) == active:
        names = {k: llm.SpeakerName(**v) for k, v in cached.items()}
    else:
        names = llm.name_speakers(client, cfg, segments, speakers, usage)
        cache.put("names", {k: v.model_dump() for k, v in names.items() if k in active})
    assign_labels(speakers, names, segments, cfg.name_threshold)

    # 8. Outputs ------------------------------------------------------------------------
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
        "voiceprints": cfg.voiceprints and voiceprint.available(), "cost": cost,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    render.write_all(out_dir, segments, speakers, meta)
    log.info("Done in %.0fs, est. cost $%.3f -> %s", meta["elapsed_seconds"], cost["usd"], out_dir)
    return out_dir


def llm_cost(model: str, inp: int, cached: int, out: int) -> float:
    p_in, p_cached, p_out = LLM_PRICE_PER_1M.get(model, (2.0, 0.2, 10.0))
    return ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6


def combine_confidence(s: Segment) -> float:
    """Blend of whatever evidence exists; each source is already in [0, 1].

    acoustic (voiceprint) 0.5, chunk linking 0.3, LLM conversation check 0.2 —
    renormalised over the sources present — then discounted for very short or very
    quiet segments, where every signal is less reliable.
    """
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


def assign_labels(speakers, names: dict, segments: list[Segment], threshold: float) -> None:
    """Real names only above `threshold`, no duplicates; everyone else is Guest N in
    order of first appearance."""
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
