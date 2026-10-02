import numpy as np
import json

import pytest

from minutesman import pipeline, voiceprint
from minutesman.config import Settings
from minutesman.models import Segment
from minutesman.speakers import SpeakerRegistry
from tests.fakes import FakeClient, synth, write_wav

TURNS = [("ali", 6), ("sara", 5), ("bilal", 7), ("ali", 4), ("sara", 6), ("ayesha", 5),
         ("ali", 5), ("bilal", 6), ("sara", 4), ("ali", 6), ("ayesha", 6), ("bilal", 5)] * 2


def cfg(**kw):
    c = Settings()
    return c.update(chunk_seconds=60, overlap_seconds=10, window_seconds=20, voiceprints=False,
                    concurrency=2, **kw)


@pytest.fixture
def meeting(tmp_path):
    pcm, truth = synth(TURNS)
    return write_wav(tmp_path / "meeting.wav", pcm), truth


def load(out):
    return json.loads((out / "transcript.json").read_text(encoding="utf-8"))


def test_end_to_end_consistent_speakers(tmp_path, meeting):
    src, truth = meeting
    client = FakeClient()
    out = pipeline.run(src, tmp_path / "out", cfg(), client=client)
    data = load(out)
    # Four real people across 3+ chunks must come out as exactly four speakers.
    assert len(data["speakers"]) == 4
    by_speaker = {}
    for s in data["segments"]:
        who = s["text"].split()[0]
        by_speaker.setdefault(s["speaker"], set()).add(who)
    assert all(len(v) == 1 for v in by_speaker.values()), by_speaker
    # Every true turn appears exactly once despite chunk overlap.
    assert len(data["segments"]) == len(truth)
    # Fusion text replaced pass A text.
    assert all("kehta hai" in s["text"] for s in data["segments"])
    # Later chunks were told about earlier speakers.
    diar = [c for c in client.audio.transcriptions.calls if c["model"].endswith("diarize")]
    assert len(diar) >= 3 and all(1 <= len(c["names"]) <= 4 for c in diar[1:])


def test_labels_names_and_guests(tmp_path, meeting):
    src, _ = meeting
    data = load(pipeline.run(src, tmp_path / "out", cfg(), client=FakeClient()))
    labels = {s["label"] for s in data["speakers"]}
    # "ali" is named confidently; "sara" only at 0.5 (< 0.75) so stays a guest.
    assert "Ali" in labels and "Sara" not in labels
    assert {"Guest 1", "Guest 2", "Guest 3"} <= labels
    sara = next(s for s in data["speakers"] if s["name_guess"] == "Sara")
    assert sara["label"].startswith("Guest")
    md = (tmp_path / "out" / "transcript.md").read_text(encoding="utf-8")
    assert "maybe Sara" in md and "**[00:00:00] Ali**" in md
    for s in data["segments"]:
        assert 0 <= s["confidence"] <= 1


def test_cache_avoids_repeat_api_calls(tmp_path, meeting):
    src, _ = meeting
    pipeline.run(src, tmp_path / "out", cfg(), client=FakeClient())
    again = FakeClient()
    out = pipeline.run(src, tmp_path / "out", cfg(), client=again)
    assert again.audio.transcriptions.calls == [] and again.responses.calls == 0
    assert load(out)["meta"]["cost"]["usd"] == 0


def test_enrolled_voice_gets_name(tmp_path, meeting):
    src, _ = meeting
    sample, _ = synth([("bilal", 6)])
    voice = write_wav(tmp_path / "bilal.wav", sample)
    data = load(pipeline.run(src, tmp_path / "out", cfg(), voices={"Bilal Khan": voice}, client=FakeClient()))
    bilal = [s for s in data["speakers"] if s["label"] == "Bilal Khan"]
    assert len(bilal) == 1 and bilal[0]["enrolled"]
    assert all(s["text"].startswith("bilal") for s in data["segments"] if s["label"] == "Bilal Khan")


def test_overlap_vote_links_unreferenced_speaker():
    reg = SpeakerRegistry(max_known=4)
    prev = [Segment("p1", 0, 50, 58, "A", "x", speaker="S1"), Segment("p2", 0, 58, 60, "B", "y", speaker="S2")]
    reg._new(), reg._new()
    cur = [Segment("c1", 1, 50.2, 57.9, "A", "x"), Segment("c2", 1, 70, 75, "B", "z")]
    reg.link_chunk(1, cur, prev, referenced=[])
    assert cur[0].speaker == "S1" and cur[0].link_confidence > 0.8
    assert cur[1].speaker == "S3"  # no evidence: new person


def test_combine_confidence_penalises_short_quiet():
    s = Segment("a", 0, 0, 5, "A", "x", link_confidence=0.85, llm_confidence=0.9, level_dbfs=-20)
    loud = pipeline.combine_confidence(s)
    s2 = Segment("b", 0, 0, 0.5, "A", "x", link_confidence=0.85, llm_confidence=0.9, level_dbfs=-50)
    assert pipeline.combine_confidence(s2) < loud


def test_voiceprint_refine_merges_split_speaker():
    import numpy as np

    class PitchEmbedder:
        def embed(self, pcm):
            from tests.fakes import FREQS, pitch
            v = np.zeros(len(FREQS), np.float32)
            v[list(FREQS).index(pitch(pcm))] = 1
            return v

    pcm, truth = synth([("ali", 3), ("sara", 3), ("ali", 3)])
    reg = SpeakerRegistry()
    for _ in range(3):
        reg._new()
    segs = [Segment(f"s{i}", 0, a, b, "A", w, speaker=f"S{i + 1}", link_confidence=0.7)
            for i, (w, a, b) in enumerate(truth)]
    for s in segs:
        reg.entries[s.speaker].speaker.talk_seconds = s.duration
    voiceprint.refine(segs, pcm, PitchEmbedder(), {}, reg)
    assert segs[0].speaker == segs[2].speaker != segs[1].speaker
    assert segs[0].acoustic_confidence > 0.9 and segs[2].acoustic_confidence > 0.9
    assert 0.5 <= segs[1].acoustic_confidence <= 0.6  # a lone segment is never certain


# Two people in one far room got one label; voiceprints split them
def test_voiceprint_moves_segment_the_diarizer_misattributed():
    import numpy as np

    class PitchEmbedder:
        def embed(self, pcm):
            from tests.fakes import FREQS, pitch
            v = np.full(len(FREQS), 0.1, np.float32)
            v[list(FREQS).index(pitch(pcm))] = 1
            return v / np.linalg.norm(v)

    pcm, truth = synth([("sara", 3), ("bilal", 3), ("sara", 3), ("bilal", 3), ("sara", 3), ("bilal", 3)])
    reg = SpeakerRegistry()
    reg._new(), reg._new()
    # Diarizer: first Sara turn right (S1); everything after is labelled S2 (Bilal).
    segs = [Segment(f"s{i}", 0, a, b, "A" if i == 0 else "B", w, speaker="S1" if i == 0 else "S2",
                    link_confidence=0.7) for i, (w, a, b) in enumerate(truth)]
    voiceprint.refine(segs, pcm, PitchEmbedder(), {}, reg)
    assert [s.speaker for s in segs] == ["S1", "S2", "S1", "S2", "S1", "S2"]
    assert "voice fits S1" in segs[2].notes


def test_short_segments_follow_their_fingerprinted_neighbour():
    from minutesman.voiceprint import _assign_short

    segs = [Segment("a", 0, 0, 1.0, "D", "hiring ke", speaker="S2"),  # too short, stale id
            Segment("b", 0, 1.2, 5.0, "D", "baare mein", speaker="S5"),  # re-clustered
            Segment("c", 0, 5.5, 6.0, "A", "ok", speaker="S1")]  # other label: untouched
    _assign_short(segs, {"b"})
    assert [s.speaker for s in segs] == ["S5", "S5", "S1"]


def test_evaluate_scores_speakers_and_text():
    import sys
    sys.path.insert(0, "scripts")
    from evaluate import evaluate

    truth = [{"speaker": "Ali", "condition": "clean", "start": 0, "end": 4, "reference": "haan theek hai"},
             {"speaker": "Sara", "condition": "far", "start": 5, "end": 9, "reference": "okay done"}]
    run = {"speakers": [{"id": "S1", "label": "Ali"}, {"id": "S2", "label": "Guest 1"}],
           "segments": [{"speaker": "S1", "start": 0, "end": 4, "text": "Haan, theek hai."},
                        {"speaker": "S1", "start": 5, "end": 9, "text": "okay done"}]}
    r = evaluate(run, truth)
    assert r["cer"] == 0.0 and r["turns_correct"] == "1/2" and r["speaker_accuracy"] == 0.5


def test_low_confidence_segment_gets_its_own_turn():
    from minutesman.render import turns

    segs = [Segment("a", 0, 0, 3, "B", "x", speaker="S2", confidence=0.8, text="nahi abhi pending hai"),
            Segment("b", 0, 3.5, 6, "B", "y", speaker="S2", confidence=0.5, text="aur client demo"),
            Segment("c", 0, 6.5, 8, "B", "z", speaker="S2", confidence=0.55, text="right?")]
    t = turns(segs)
    assert [x["text"] for x in t] == ["nahi abhi pending hai", "aur client demo right?"]
    assert t[1]["confidence"] < 0.6


def test_split_meetings_cleans_llm_ranges():
    from minutesman import llm

    segs = [Segment(f"s{i}", 0, i * 10, i * 10 + 5, "A", "x", speaker="S1" if i < 4 else "S2")
            for i in range(8)]
    proposed = [llm.Meeting(first_line=4, last_line=9, title="Hiring", boundary_evidence=""),
                llm.Meeting(first_line=0, last_line=4, title="Budget", boundary_evidence="")]  # overlaps
    meetings, index_of = pipeline.split_meetings(segs, proposed)
    assert [m["title"] for m in meetings] == ["Budget", "Hiring"] and index_of[0] == 1
    assert [s.meeting for s in segs] == [0, 0, 0, 0, 0, 1, 1, 1]
    assert meetings[1]["participants"] == ["S2"] and meetings[1]["end"] == 75


def test_split_meetings_leaves_hallway_outside():
    from minutesman import llm

    segs = [Segment(f"s{i}", 0, i * 10, i * 10 + 5, "A", "x", speaker="S1") for i in range(6)]
    proposed = [llm.Meeting(first_line=0, last_line=1, title="A", boundary_evidence=""),
                llm.Meeting(first_line=4, last_line=5, title="B", boundary_evidence="")]
    pipeline.split_meetings(segs, proposed)
    assert [s.meeting for s in segs] == [0, 0, -1, -1, 1, 1]


# The fake analyst starts a new meeting at a 'salam' line
def test_two_meetings_end_to_end(tmp_path):
    turns = [("ali", 5), ("sara", 5), ("ali", 4), ("bilal", 5), ("ayesha", 5), ("ali", 5), ("ayesha", 4)]
    pcm, _ = synth(turns)
    src = write_wav(tmp_path / "two.wav", pcm)

    class Salam(FakeClient):
        def __init__(self):
            super().__init__()
            orig = self.audio.transcriptions.create

            def create(*a, **kw):
                r = orig(*a, **kw)
                events = list(r) if kw.get("stream") else getattr(r, "segments", [])
                for s in events:
                    if getattr(s, "text", "").startswith("ayesha") and s.start < 30:
                        s.text = "ayesha salam everyone"
                return iter(events) if kw.get("stream") else r
            self.audio.transcriptions.create = create

    out = pipeline.run(src, tmp_path / "out", cfg().update(chunk_seconds=600), client=Salam())
    data = load(out)
    assert len(data["meta"]["meetings"]) == 2
    md = (out / "transcript.md").read_text(encoding="utf-8")
    assert "## Meetings" in md and "## Meeting 2: M1" in md


def test_same_id_named_differently_in_two_meetings_is_split():
    from minutesman import llm

    reg = SpeakerRegistry()
    sp = reg._new().speaker
    segs = [Segment("a", 0, 0, 10, "B", "x", speaker=sp.id, meeting=0),
            Segment("b", 0, 60, 64, "B", "y", speaker=sp.id, meeting=1)]
    entries = [llm.SpeakerName(speaker=sp.id, meeting=0, name="Sara", confidence=0.9, evidence=""),
               llm.SpeakerName(speaker=sp.id, meeting=1, name="Ayesha", confidence=0.85, evidence="")]
    speakers = [sp]
    best = pipeline.resolve_names(segs, speakers, entries, {0: 0, 1: 1, -1: -1}, reg, 0.75)
    assert segs[0].speaker == sp.id and segs[1].speaker != sp.id
    assert best[sp.id].name == "Sara" and best[segs[1].speaker].name == "Ayesha"
    assert len(speakers) == 2


def test_weak_second_name_does_not_split():
    from minutesman import llm

    reg = SpeakerRegistry()
    sp = reg._new().speaker
    segs = [Segment("a", 0, 0, 10, "B", "x", speaker=sp.id, meeting=0),
            Segment("b", 0, 60, 64, "B", "y", speaker=sp.id, meeting=1)]
    entries = [llm.SpeakerName(speaker=sp.id, meeting=0, name="Sara", confidence=0.9, evidence=""),
               llm.SpeakerName(speaker=sp.id, meeting=1, name="Ayesha", confidence=0.4, evidence="")]
    best = pipeline.resolve_names(segs, [sp], entries, {0: 0, 1: 1, -1: -1}, reg, 0.75)
    assert segs[1].speaker == sp.id and best[sp.id].name == "Sara"


def test_hedged_request_takes_the_faster_copy():
    import time
    from minutesman.progress import hedged

    calls = []

    def call():
        calls.append(1)
        time.sleep(2 if len(calls) == 1 else 0.1)
        return len(calls)

    assert hedged(call, 0.3, "t") == 2


def test_old_cache_without_span_is_still_used(tmp_path, meeting):
    import json as js

    src, _ = meeting
    pipeline.run(src, tmp_path / "out", cfg(), client=FakeClient())
    for f in (tmp_path / "out" / "work" / "cache").glob("passA_*.json"):
        d = js.loads(f.read_text())
        d.pop("span")
        f.write_text(js.dumps(d))
    again = FakeClient()
    pipeline.run(src, tmp_path / "out", cfg(), client=again)
    assert not any(c["model"].endswith("diarize") for c in again.audio.transcriptions.calls)


def test_ids_never_reused_after_a_drop():
    reg = SpeakerRegistry()
    a, b = reg._new().speaker.id, reg._new().speaker.id
    reg.entries.pop(a)
    c = reg._new().speaker.id
    assert c not in (a, b) and c in reg.entries and b in reg.entries
    assert reg.enroll("Raja", np.zeros(16000, np.float32)).startswith("E")


def test_fragment_speakers_fold_into_neighbours():
    segs = [Segment("a", 0, 0, 200, "A", "x", speaker="S1", confidence=0.9),
            Segment("b", 0, 201, 202, "A", "mm-hmm", speaker="S7", confidence=0.5),
            Segment("c", 0, 203, 400, "B", "y", speaker="S2", confidence=0.9),
            Segment("d", 0, 401, 402, "B", "yeah", speaker="S8", confidence=0.6),
            Segment("e", 0, 403, 415, "C", "short but real", speaker="S9", confidence=0.8)]
    pipeline.absorb_minor_speakers(segs, 15)
    assert [s.speaker for s in segs] == ["S1", "S1", "S2", "S2", "S9"]
    assert "folded" in segs[1].notes


def test_rename_overrides_labels(tmp_path, meeting):
    src, _ = meeting
    data = load(pipeline.run(src, tmp_path / "out", cfg(rename={"S2": "Raja"}), client=FakeClient()))
    assert any(sp["id"] == "S2" and sp["label"] == "Raja" for sp in data["speakers"])


def test_off_meeting_lines_are_hidden_in_clean_transcript(tmp_path):
    from minutesman import llm, render
    from minutesman.models import Speaker

    segs = [Segment(f"s{i}", 0, i * 10, i * 10 + 5, "A", "x", speaker="S1", confidence=0.9,
                    text=t) for i, t in enumerate(["budget review", "haan ji, Teams pe call hai", "wapis aa gaya"])]
    spans = pipeline.mark_off_meeting(segs, [llm.OffMeeting(first_line=1, last_line=1, reason="side Teams call")])
    meetings, _ = pipeline.split_meetings(segs, [llm.Meeting(first_line=0, last_line=2, title="T",
                                                             boundary_evidence="")])
    meta = {"source": "x.m4a", "duration_seconds": 30, "cost": {"usd": 0}, "meetings": meetings, "off_meeting": spans}
    render.write_all(tmp_path, segs, [Speaker(id="S1", label="Raja", talk_seconds=15)], meta)
    clean = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    full = (tmp_path / "transcript_full.md").read_text(encoding="utf-8")
    assert "Teams pe call" not in clean and "left out: side Teams call" in clean
    assert "Teams pe call" in full and "budget review" in clean and "wapis aa gaya" in clean
    assert segs[1].meeting == -1 and segs[2].meeting == 0


def test_trim_keeps_original_timestamps(tmp_path, meeting):
    src, _ = meeting
    client = FakeClient()
    data = load(pipeline.run(src, tmp_path / "out", cfg(trim_start=20.0, trim_end=80.0), client=client))
    assert data["segments"] and min(s["start"] for s in data["segments"]) >= 20.0
    assert max(s["end"] for s in data["segments"]) <= 80.5
