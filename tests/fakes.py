# Offline OpenAI stand-in: synthetic speakers are tones, recognised by pitch
from __future__ import annotations

import base64
import string
from types import SimpleNamespace

import numpy as np

from minutesman import audio, llm

SR = audio.SAMPLE_RATE
FREQS = {"ali": 220.0, "sara": 390.0, "bilal": 560.0, "ayesha": 760.0}


# Tone 'speech' for (speaker, seconds) turns; returns pcm and truth
def synth(turns: list[tuple[str, float]], gap: float = 0.6, level: float = 0.3) -> tuple[np.ndarray, list]:
    pcm, truth, t = [], [], 0.0
    for spk, dur in turns:
        n = int(dur * SR)
        x = np.arange(n) / SR
        tone = level * (np.sin(2 * np.pi * FREQS[spk] * x) + 0.3 * np.sin(4 * np.pi * FREQS[spk] * x))
        pcm += [tone.astype(np.float32), np.zeros(int(gap * SR), np.float32)]
        truth.append((spk, t, t + dur))
        t += dur + gap
    return np.concatenate(pcm), truth


def pitch(pcm: np.ndarray) -> str:
    spec = np.abs(np.fft.rfft(pcm[: SR * 4] * np.hanning(min(len(pcm), SR * 4))))
    f = np.fft.rfftfreq(min(len(pcm), SR * 4), 1 / SR)[np.argmax(spec)]
    return min(FREQS, key=lambda k: abs(FREQS[k] - f))


def bursts(pcm: np.ndarray) -> list[tuple[float, float]]:
    frame = SR // 50
    e = np.sqrt(np.mean(pcm[: len(pcm) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    active = e > 0.02
    runs, start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if (i - start) * 0.02 > 0.3:
                runs.append((start * 0.02, i * 0.02))
            start = None
    return runs


def _decode(data: bytes) -> np.ndarray:
    raw = audio._run(["-i", "-", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], input_bytes=data)
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


class FakeTranscriptions:
    def __init__(self):
        self.calls = []

    def create(self, model, file, known_speaker_names=None, known_speaker_references=None, **kw):
        name, fh, _ = file
        pcm = _decode(fh.read())
        self.calls.append({"model": model, "names": list(known_speaker_names or []), **kw})
        if model.endswith("diarize"):
            known = {}
            for sid, ref in zip(known_speaker_names or [], known_speaker_references or []):
                ref_pcm = _decode(base64.b64decode(ref.split(",", 1)[1]))
                known[pitch(ref_pcm)] = sid
            letters, segs = {}, []
            for a, b in bursts(pcm):
                who = pitch(audio.slice_pcm(pcm, a, b))
                label = known.get(who) or letters.setdefault(who, string.ascii_uppercase[len(letters)])
                segs.append(SimpleNamespace(speaker=label, start=a, end=b, text=f"{who} bolta hai"))
            return SimpleNamespace(segments=segs)
        who = [pitch(audio.slice_pcm(pcm, a, b)) for a, b in bursts(pcm)]
        return SimpleNamespace(text=" ".join(f"{w} says" for w in who), languages=[{"code": "ur"}])


class FakeResponses:
    def __init__(self):
        self.calls = 0

    def parse(self, model, input, text_format, **kw):
        self.calls += 1
        import json

        usage = SimpleNamespace(input_tokens=1000, output_tokens=500,
                                input_tokens_details=SimpleNamespace(cached_tokens=0))
        user = input[-1]["content"]
        if text_format is llm.FusedChunk:
            data = json.loads(user)
            segs = [llm.FusedSegment(id=s["id"], text=s["text"].replace("bolta hai", "kehta hai"),
                                     text_confidence=0.9, speaker_confidence=0.8,
                                     suggested_speaker=None, note="")
                    for w in data["windows"] for s in w["pass_a_segments"]]
            return SimpleNamespace(output_parsed=llm.FusedChunk(segments=segs), usage=usage)
        # Names "ali" confidently and "sara" weakly; a "salam" line starts a new meeting
        ids = user.split("Speaker ids: ")[1].split("\n")[0].split(", ")
        lines = [ln for ln in user.split("Transcript:\n")[1].splitlines() if ln.startswith("L")]
        first, starts = {}, [0]
        for ln in lines:
            num = int(ln.split(" ", 1)[0][1:])
            sid, text = ln.split("] ", 1)[1].split(": ", 1)
            first.setdefault(sid, text.split()[0])
            if "salam" in text and num > 0:
                starts.append(num)
        out = []
        for sid in ids:
            who = first.get(sid)
            conf = {"ali": 0.9, "sara": 0.5}.get(who, 0.1)
            out.append(llm.SpeakerName(speaker=sid, meeting=0, name=who.title() if conf > 0.3 else None,
                                       confidence=conf, evidence="test"))
        ends = [s - 1 for s in starts[1:]] + [len(lines) - 1]
        meetings = [llm.Meeting(first_line=a, last_line=b, title=f"M{k}", boundary_evidence="test")
                    for k, (a, b) in enumerate(zip(starts, ends))]
        return SimpleNamespace(output_parsed=llm.Analysis(speakers=out, meetings=meetings), usage=usage)


class FakeClient:
    def __init__(self):
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())
        self.responses = FakeResponses()


def write_wav(path, pcm):
    path.write_bytes(audio.pcm_to_wav_bytes(pcm))
    return path
