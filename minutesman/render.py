# Writes transcript.md, .txt, .srt and .json
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import Segment, Speaker

LOW_CONFIDENCE = 0.6
GAP_NOTE_SECONDS = 30


def ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


# Merge same-speaker segments into turns; doubtful segments stay separate so the warning shows
def turns(segments: list[Segment], max_gap: float = 2.0) -> list[dict]:
    out: list[dict] = []
    prev_low = False
    for s in segments:
        last = out[-1] if out else None
        low = s.confidence < LOW_CONFIDENCE
        if (last and last["speaker"] == s.speaker and last["meeting"] == s.meeting
                and last["off"] == s.off_reason and s.start - last["end"] <= max_gap and low == prev_low):
            last["end"] = s.end
            last["texts"].append(s.text)
            last["confs"].append((s.confidence, s.duration))
        else:
            out.append({"speaker": s.speaker, "meeting": s.meeting, "off": s.off_reason, "start": s.start, "end": s.end,
                        "texts": [s.text], "confs": [(s.confidence, s.duration)]})
        prev_low = low
    for t in out:
        w = sum(d for _, d in t["confs"]) or 1.0
        t["confidence"] = round(sum(c * d for c, d in t["confs"]) / w, 2)
        t["text"] = " ".join(x for x in t.pop("texts") if x)
        del t["confs"]
    return out


# Transcript sections; full=False collapses off-meeting stretches into one marker line each
def _body(tt: list[dict], meetings: list[dict], label: dict, full: bool) -> list[str]:
    md: list[str] = []
    if len(meetings) <= 1:
        md += ["## Transcript", ""]
    current, prev_end, hidden = None, None, []

    def flush():
        if hidden:
            md.extend([f"*⋯ {ts(hidden[0]['start'])}–{ts(hidden[-1]['end'])} left out: {hidden[0]['off']} "
                       f"({len(hidden)} turns, see transcript_full.md)*", ""])
            hidden.clear()

    for t in tt:
        if t["off"] and not full:
            if hidden and hidden[-1]["off"] != t["off"]:
                flush()
            hidden.append(t)
            prev_end = t["end"]
            continue
        flush()
        section = "off" if t["off"] else t["meeting"]
        if len(meetings) > 1 or t["off"] or current == "off":
            if section != current:
                current = section
                if t["off"]:
                    md += [f"## Off-meeting: {t['off']}", ""]
                elif current >= 0:
                    m = meetings[current]
                    md += [f"## Meeting {current + 1}: {m['title']} ({ts(m['start'])}–{ts(m['end'])})", ""]
                else:
                    md += ["## Between meetings", ""]
        if prev_end is not None and t["start"] - prev_end >= GAP_NOTE_SECONDS:
            md += [f"*… {(t['start'] - prev_end) / 60:.1f} min without speech …*", ""]
        prev_end = t["end"]
        flag = " ⚠" if t["confidence"] < LOW_CONFIDENCE else ""
        md.append(f"**[{ts(t['start'])}] {label.get(t['speaker'], t['speaker'])}** "
                  f"({t['confidence']:.2f}){flag}: {t['text']}")
        md.append("")
    flush()
    return md


def write_all(out_dir: Path, segments: list[Segment], speakers: list[Speaker], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = {sp.id: sp.label for sp in speakers}
    tt = turns(segments)

    # Markdown
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
    meetings = meta.get("meetings") or []
    if len(meetings) > 1:
        md += ["", "## Meetings", "", "| # | Time | Topic | Participants |", "|---|---|---|---|"]
        for m in meetings:
            who = ", ".join(label.get(p, p) for p in m["participants"])
            md.append(f"| {m['index'] + 1} | {ts(m['start'])}–{ts(m['end'])} | {m['title']} | {who} |")
    off = meta.get("off_meeting") or []
    if off:
        md += ["", "## Left out of the transcript", "",
               "Not part of any meeting; the full text is in transcript_full.md.", "",
               "| Time | Why |", "|---|---|"]
        md += [f"| {ts(o['start'])}–{ts(o['end'])} | {o['reason']} |" for o in off]
    md += ["", "Speaker confidence in brackets; ⚠ marks turns below "
           f"{LOW_CONFIDENCE:.0%}.", ""]
    header = list(md)
    clean = _body(tt, meetings, label, full=False)
    (out_dir / "transcript.md").write_text("\n".join(header + clean), encoding="utf-8")
    (out_dir / "transcript_full.md").write_text("\n".join(header + _body(tt, meetings, label, full=True)),
                                               encoding="utf-8")
    tt_clean = [t for t in tt if not t["off"]]

    # Plain text
    (out_dir / "transcript.txt").write_text(
        "\n".join(f"[{ts(t['start'])}] {label.get(t['speaker'], t['speaker'])} ({t['confidence']:.2f}): "
                  f"{t['text']}" for t in tt_clean) + "\n", encoding="utf-8")

    # SRT
    srt = []
    for i, t in enumerate(tt, 1):
        srt += [str(i), f"{ts(t['start'], True)} --> {ts(t['end'], True)}",
                f"{label.get(t['speaker'], t['speaker'])}: {t['text']}", ""]
    (out_dir / "transcript.srt").write_text("\n".join(srt), encoding="utf-8")

    # JSON
    data = {
        "meta": meta,
        "speakers": [asdict(sp) for sp in speakers],
        "turns": [{**t, "label": label.get(t["speaker"], t["speaker"])} for t in tt],
        "segments": [{**asdict(s), "label": label.get(s.speaker, s.speaker)} for s in segments],
    }
    (out_dir / "transcript.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
