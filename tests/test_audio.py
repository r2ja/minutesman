import numpy as np
import pytest

from minutesman import audio
from tests.fakes import synth, write_wav


def test_plan_chunks_covers_everything_once():
    chunks = audio.plan_chunks(4800, 600, 30)
    assert chunks[0].start == 0 and chunks[-1].end == 4800
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlap
        assert a.keep_end == b.keep_start  # ownership is contiguous
    assert all(c.end - c.start <= 600 + 60 for c in chunks)


def test_plan_chunks_absorbs_tiny_tail():
    chunks = audio.plan_chunks(1150, 600, 30)
    assert len(chunks) == 2 and chunks[-1].end == 1150


def test_plan_chunks_short_file():
    [c] = audio.plan_chunks(42.0, 600, 30)
    assert (c.start, c.end, c.keep_start, c.keep_end) == (0, 42.0, 0, 42.0)


def test_plan_chunks_rejects_bad_overlap():
    with pytest.raises(ValueError):
        audio.plan_chunks(100, 30, 30)


@pytest.mark.parametrize("enhance", ["off", "light", "strong", "rnnoise"])
def test_preprocess_roundtrip(tmp_path, enhance):
    pcm, _ = synth([("ali", 2.0), ("sara", 2.0)])
    src = write_wav(tmp_path / "in.wav", pcm)
    out = audio.preprocess(src, tmp_path / "out.wav", enhance)
    assert abs(audio.duration(out) - len(pcm) / audio.SAMPLE_RATE) < 0.1
    assert audio.load_pcm(out).size > 0


# A far talker at about -46 dBFS must come out usable
@pytest.mark.parametrize("enhance", ["light", "strong"])
def test_enhance_lifts_quiet_speech(tmp_path, enhance):
    pcm, _ = synth([("ali", 4.0), ("sara", 4.0)], level=0.01, gap=1.5)
    t = np.arange(len(pcm)) / audio.SAMPLE_RATE
    pcm = pcm * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)).astype(np.float32)  # syllable rhythm
    pcm += np.random.default_rng(0).normal(0, 0.0008, len(pcm)).astype(np.float32)
    src = write_wav(tmp_path / "in.wav", pcm)
    out = audio.load_pcm(audio.preprocess(src, tmp_path / "out.wav", enhance))
    assert audio.level_dbfs(out) > audio.level_dbfs(pcm) + 20


def test_mp3_is_compact():
    pcm = np.zeros(audio.SAMPLE_RATE * 60, np.float32)
    assert len(audio.pcm_to_mp3_bytes(pcm)) < 600_000  # 1 min at 64 kbps ~ 480 KB
