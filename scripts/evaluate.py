# Score a run against make_test_audio.py truth: speaker accuracy, meetings and text CER
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
    meetings = {}
    if "meeting" in truth[0]:
        # Each true turn gets the meeting of the output segments overlapping it most
        got_m = []
        for t in truth:
            votes: dict[int, float] = defaultdict(float)
            for s in segs:
                ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
                if ov > 0:
                    votes[s.get("meeting", -1)] += ov
            got_m.append(max(votes, key=votes.get) if votes else -1)
        pairs = defaultdict(lambda: defaultdict(int))
        for t, g in zip(truth, got_m):
            pairs[g][t["meeting"]] += 1
        m_map = {g: max(v, key=v.get) for g, v in pairs.items()}
        right = sum(m_map[g] == t["meeting"] for t, g in zip(truth, got_m))
        meetings = {"meetings_found": len(run["meta"].get("meetings", [])),
                    "meetings_true": len({t["meeting"] for t in truth if t["meeting"] >= 0}),
                    "turns_in_right_meeting": f"{right}/{len(truth)}"}
    return {
        **meetings,
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
