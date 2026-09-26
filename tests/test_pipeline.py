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
    pipeline.run(src, tmp_path / "out", cfg(), client=again)
    assert again.audio.transcriptions.calls == [] and again.responses.calls == 0


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


def test_voiceprint_moves_segment_the_diarizer_misattributed():
    """Two people in the same far-away room got one label; the voice says otherwise."""
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
