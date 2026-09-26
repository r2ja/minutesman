"""Score a minutesman run against a ground-truth file from make_test_audio.py.

    python scripts/evaluate.py output/synth/transcript.json samples/synthetic_meeting.truth.json

Speaker accuracy: share of reference speech time whose output speaker maps to the right
person. Each output speaker is mapped to the true person it overlaps most, so the score
reflects diarization, not naming. Several output ids can map to one person; each extra
id is reported as a split.
Text: character error rate (CER) of the whole transcript against the Roman Urdu
reference, after lowercasing and stripping punctuation. CER is used instead of WER
because Roman Urdu spelling varies ("hai"/"hay", "nahi"/"nahin").
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict


def _norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    hyp, ref = _norm(hyp), _norm(ref)
    return _edit_distance(hyp, ref) / max(1, len(ref))


def evaluate(run: dict, truth: list[dict]) -> dict:
    segs = run["segments"]
    overlap: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for s in segs:
        for t in truth:
            ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if ov > 0:
                overlap[s["speaker"]][t["speaker"]] += ov
    mapping = {spk: max(v, key=v.get) for spk, v in overlap.items()}
    correct = sum(v[mapping[spk]] for spk, v in overlap.items())
    total = sum(t["end"] - t["start"] for t in truth)
    per_turn = []
    for t in truth:
        votes: dict[str, float] = defaultdict(float)
        for s in segs:
            ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if ov > 0:
                votes[mapping[s["speaker"]]] += ov
        got = max(votes, key=votes.get) if votes else "-"
        per_turn.append((t["condition"], t["speaker"], got))
    people = {t["speaker"] for t in truth}
    by_cond: dict[str, list[bool]] = defaultdict(list)
    for cond, want, got in per_turn:
        by_cond[cond].append(want == got)
    hyp = " ".join(s["text"] for s in segs)
    ref = " ".join(t["reference"] for t in truth)
    return {
        "speaker_accuracy": round(correct / total, 3),
        "turns_correct": f"{sum(w == g for _, w, g in per_turn)}/{len(per_turn)}",
        "turns_correct_by_condition": {c: f"{sum(v)}/{len(v)}" for c, v in by_cond.items()},
        "speakers_found": len(overlap),
        "speakers_true": len(people),
        "cer": round(cer(hyp, ref), 3),
        "labels": {spk: next((x["label"] for x in run["speakers"] if x["id"] == spk), spk) + f" = {who}"
                   for spk, who in mapping.items()},
    }


def main() -> None:
    run = json.load(open(sys.argv[1], encoding="utf-8"))
    truth = json.load(open(sys.argv[2], encoding="utf-8"))
    print(json.dumps(evaluate(run, truth), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
