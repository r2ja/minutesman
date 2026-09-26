"""Optional local speaker embeddings (SpeechBrain ECAPA-TDNN, CPU is fine).

Install with `pip install -e ".[voiceprint]"`. When present, they:
- merge global speakers the API linking split apart (same voice, two ids),
- give every segment an acoustic confidence (how close its voice is to its speaker),
- flag segments whose voice matches a different speaker much better,
- match speakers to user-supplied voice samples beyond the API's 4-reference limit.
Without the extra, the pipeline still runs; confidences then come from linking + LLM only.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, slice_pcm
from .models import Segment

log = logging.getLogger(__name__)

MIN_SECONDS = 1.0
MERGE_SIMILARITY = 0.72  # centroids at/above this are treated as one person...
# ...unless the diarizer heard both in the same chunk and kept them apart: overriding
# it then needs near-identical voices.
MERGE_SIMILARITY_SAME_CHUNK = 0.86
REASSIGN_MARGIN = 0.25


def available() -> bool:
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class Embedder:
    def __init__(self, cache_dir: Path):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self._torch = torch
        # COPY avoids symlinks, which fail on Windows without developer mode.
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(cache_dir / "ecapa"),
            run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY,
        )

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        pcm = pcm[: 15 * SAMPLE_RATE]
        with self._torch.no_grad():
            e = self.model.encode_batch(self._torch.from_numpy(pcm.copy()).unsqueeze(0))
        v = e.squeeze().cpu().numpy().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


def _centroid(vecs: list[np.ndarray], weights: list[float]) -> np.ndarray:
    c = np.average(np.stack(vecs), axis=0, weights=weights)
    return c / (np.linalg.norm(c) + 1e-9)


def refine(segs: list[Segment], pcm: np.ndarray, embedder: Embedder,
           enrolled: dict[str, np.ndarray], registry) -> None:
    """Refine speaker ids in place. `enrolled` maps enrolled speaker id -> sample embedding;
    each sample counts as 10 s of that speaker's speech, so matching voices merge into it."""
    embs: dict[str, np.ndarray] = {}
    for s in segs:
        if s.duration >= MIN_SECONDS:
            embs[s.id] = embedder.embed(slice_pcm(pcm, s.start, s.end))

    def centroids() -> dict[str, np.ndarray]:
        groups: dict[str, tuple[list, list]] = {
            sid: ([e], [10.0]) for sid, e in enrolled.items() if sid in registry.entries
        }
        for s in segs:
            if s.id in embs:
                g = groups.setdefault(s.speaker, ([], []))
                g[0].append(embs[s.id])
                g[1].append(s.duration)
        return {k: _centroid(v, w) for k, (v, w) in groups.items()}

    # 1. Merge speakers whose voices are near-identical (greedy, most similar pair first).
    while True:
        cents = centroids()
        ids = sorted(cents)
        chunks_of = {k: {s.chunk for s in segs if s.speaker == k} for k in ids}
        pairs = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if registry.entries[a].speaker.enrolled and registry.entries[b].speaker.enrolled:
                    continue  # two distinct voice samples: trust the user
                bar = MERGE_SIMILARITY_SAME_CHUNK if chunks_of[a] & chunks_of[b] else MERGE_SIMILARITY
                sim = float(cents[a] @ cents[b])
                if sim >= bar:
                    pairs.append((sim - bar, sim, a, b))
        if not pairs:
            break
        _, sim, a, b = max(pairs)
        ta, tb = (registry.entries[x].speaker for x in (a, b))
        if ta.enrolled or (not tb.enrolled and ta.talk_seconds >= tb.talk_seconds):
            keep, drop = a, b
        else:
            keep, drop = b, a
        log.info("voiceprint: merging %s into %s (similarity %.2f)", drop, keep, sim)
        registry.merge(keep, drop, segs)

    # 2. Per-segment acoustic confidence; move clear outliers to the better speaker.
    cents = centroids()
    for s in segs:
        v = embs.get(s.id)
        if v is None or s.speaker not in cents:
            continue
        own = float(v @ cents[s.speaker])
        others = [(float(v @ c), k) for k, c in cents.items() if k != s.speaker]
        alt_sim, alt = max(others) if others else (-1.0, None)
        if alt and alt_sim - own > REASSIGN_MARGIN and s.duration >= 2.0:
            s.notes = (s.notes + f" voice matches {alt} better ({alt_sim:.2f} vs {own:.2f});").strip()
            s.speaker, own, alt_sim = alt, alt_sim, own
        margin = own - max(alt_sim, 0.0)
        s.acoustic_confidence = round(float(np.clip(0.5 + 0.6 * margin + 0.3 * (own - 0.5), 0.05, 0.99)), 3)
