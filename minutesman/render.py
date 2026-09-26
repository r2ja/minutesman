"""Writes transcript.md (readable), transcript.json (everything), transcript.srt,
transcript.txt (plain) into the output folder."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import Segment, Speaker

LOW_CONFIDENCE = 0.6


def ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


def turns(segments: list[Segment], max_gap: float = 2.0) -> list[dict]:
    """Merge consecutive segments by the same speaker into readable turns."""
    out: list[dict] = []
    for s in segments:
        last = out[-1] if out else None
        if last and last["speaker"] == s.speaker and s.start - last["end"] <= max_gap:
            last["end"] = s.end
            last["texts"].append(s.text)
            last["confs"].append((s.confidence, s.duration))
        else:
            out.append({"speaker": s.speaker, "start": s.start, "end": s.end,
                        "texts": [s.text], "confs": [(s.confidence, s.duration)]})
    for t in out:
        w = sum(d for _, d in t["confs"]) or 1.0
        t["confidence"] = round(sum(c * d for c, d in t["confs"]) / w, 2)
        t["text"] = " ".join(x for x in t.pop("texts") if x)
        del t["confs"]
    return out


def write_all(out_dir: Path, segments: list[Segment], speakers: list[Speaker], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = {sp.id: sp.label for sp in speakers}
    tt = turns(segments)

    # Markdown -----------------------------------------------------------------------
    md = [f"# Transcript: {Path(meta['source']).name}", "",
          f"Duration {ts(meta['duration_seconds'])} · {len(speakers)} speakers · "
          f"est. API cost ${meta['cost']['usd']:.2f}", "", "## Speakers", "",
          "| Label | Id | Talk time | Name evidence |", "|---|---|---|---|"]
    for sp in sorted(speakers, key=lambda x: -x.talk_seconds):
        ev = sp.name_evidence.replace("|", "/").replace("\n", " ")[:160]
        if sp.name_guess and sp.label != sp.name_guess:
            ev = f"maybe {sp.name_guess} ({sp.name_confidence:.2f}, below threshold): {ev}"
        elif sp.name_guess:
            ev = f"{sp.name_confidence:.2f}: {ev}"
        md.append(f"| **{sp.label}** | {sp.id} | {ts(sp.talk_seconds)} | {ev or '-'} |")
    md += ["", "Speaker confidence in brackets; ⚠ marks turns below "
           f"{LOW_CONFIDENCE:.0%}.", "", "## Transcript", ""]
    for t in tt:
        flag = " ⚠" if t["confidence"] < LOW_CONFIDENCE else ""
        md.append(f"**[{ts(t['start'])}] {label.get(t['speaker'], t['speaker'])}** "
                  f"({t['confidence']:.2f}){flag}: {t['text']}")
        md.append("")
    (out_dir / "transcript.md").write_text("\n".join(md), encoding="utf-8")

    # Plain text -----------------------------------------------------------------------
    (out_dir / "transcript.txt").write_text(
        "\n".join(f"[{ts(t['start'])}] {label.get(t['speaker'], t['speaker'])} ({t['confidence']:.2f}): "
                  f"{t['text']}" for t in tt) + "\n", encoding="utf-8")

    # SRT ------------------------------------------------------------------------------
    srt = []
    for i, t in enumerate(tt, 1):
        srt += [str(i), f"{ts(t['start'], True)} --> {ts(t['end'], True)}",
                f"{label.get(t['speaker'], t['speaker'])}: {t['text']}", ""]
    (out_dir / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")

    # JSON -----------------------------------------------------------------------------
    data = {
        "meta": meta,
        "speakers": [asdict(sp) for sp in speakers],
        "turns": [{**t, "label": label.get(t["speaker"], t["speaker"])} for t in tt],
        "segments": [{**asdict(s), "label": label.get(s.speaker, s.speaker)} for s in segments],
    }
    (out_dir / "transcript.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
