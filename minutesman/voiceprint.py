# Optional ECAPA voiceprints: re-cluster segments by voice to fix speakers the diarizer merged by room
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, slice_pcm
from .models import Segment

log = logging.getLogger(__name__)

MIN_SECONDS = 1.5  # shorter clips give unreliable embeddings
LINK_THRESHOLD = 0.30  # stop merging clusters whose average similarity is below this
PRIOR_BONUS = 0.10  # added when chunk linking already gave two segments the same id


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


# Average-linkage clustering on a similarity matrix
def cluster(sims: np.ndarray, threshold: float, cannot_link: np.ndarray | None = None) -> list[list[int]]:
    n = len(sims)
    m = sims.astype(np.float64).copy()
    if cannot_link is not None:
        m[cannot_link] = -np.inf
    np.fill_diagonal(m, -np.inf)
    alive = np.ones(n, bool)
    size = np.ones(n)
    members = [[i] for i in range(n)]
    while alive.sum() > 1:
        masked = np.where(alive[:, None] & alive[None, :], m, -np.inf)
        a, b = np.unravel_index(np.argmax(masked), masked.shape)
        if masked[a, b] < threshold:
            break
        merged = (size[a] * m[a] + size[b] * m[b]) / (size[a] + size[b])
        m[a, :] = merged
        m[:, a] = merged
        m[a, a] = -np.inf
        alive[b] = False
        size[a] += size[b]
        members[a] += members[b]
    return [members[i] for i in range(n) if alive[i]]


# Re-assign speaker ids by voice; enrolled samples are fixed anchors that never merge
def refine(segs: list[Segment], pcm: np.ndarray, embedder: Embedder,
           enrolled: dict[str, np.ndarray], registry) -> None:
    items = [s for s in segs if s.duration >= MIN_SECONDS]
    anchors = [(sid, e) for sid, e in enrolled.items() if sid in registry.entries]
    if not items:
        return
    vecs = [embedder.embed(slice_pcm(pcm, s.start, s.end)) for s in items] + [e for _, e in anchors]
    emb = np.stack(vecs)
    sims = emb @ emb.T
    n = len(items)
    prior = [s.speaker for s in items] + [sid for sid, _ in anchors]
    weight = [s.duration for s in items] + [30.0] * len(anchors)  # an anchor outweighs any turn
    same = np.array([[a == b for b in prior] for a in prior])
    is_anchor = np.array([i >= n for i in range(len(prior))])
    cannot = is_anchor[:, None] & is_anchor[None, :] & ~same
    clusters = cluster(sims + PRIOR_BONUS * same, LINK_THRESHOLD, cannot)

    # Map clusters to ids by largest talk-time overlap; a cluster left over is a new person
    votes = []
    for c, mem in enumerate(clusters):
        tally: dict[str, float] = {}
        for i in mem:
            tally[prior[i]] = tally.get(prior[i], 0.0) + weight[i]
        votes += [(w, c, sid) for sid, w in tally.items()]
    target: dict[int, str] = {}
    used: set[str] = set()
    for _, c, sid in sorted(votes, reverse=True):
        if c not in target and sid not in used:
            target[c] = sid
            used.add(sid)
    for c in range(len(clusters)):
        if c not in target:
            target[c] = registry._new().speaker.id
            log.info("voiceprint: found an extra speaker %s the diarizer had merged", target[c])

    label = [""] * len(prior)
    for c, mem in enumerate(clusters):
        for i in mem:
            label[i] = target[c]
    for i, s in enumerate(items):
        if label[i] != s.speaker:
            log.info("voiceprint: %.1fs %s -> %s", s.start, s.speaker, label[i])
            s.notes = (s.notes + f" voice fits {label[i]} better than {s.speaker};").strip()
            s.speaker = label[i]
    _assign_short(segs, {s.id for s in items})

    # Acoustic confidence: how much better a segment fits its own cluster than the next best.
    for c, mem in enumerate(clusters):
        for i in mem:
            if i >= n:
                continue
            others = [j for j in mem if j != i]
            alt = max((sims[i, m2].mean() for d, m2 in enumerate(clusters) if d != c), default=0.0)
            if not others:
                # Only segment of its speaker: all we know is how unlike everyone else it is.
                conf = np.clip(0.65 - alt, 0.05, 0.6)
            else:
                own = sims[i, others].mean()
                conf = np.clip(0.5 + 1.5 * (own - alt) + 0.5 * (own - 0.35), 0.05, 0.99)
            items[i].acoustic_confidence = round(float(conf), 3)

    # Drop ids that lost all their speech (their segments were someone else's).
    active = {s.speaker for s in segs}
    for sid in [k for k, e in registry.entries.items() if k not in active and not e.speaker.enrolled]:
        registry.entries.pop(sid)


# Segments too short to embed follow their nearest same-label neighbour
def _assign_short(segs: list[Segment], embedded: set[str], reach: float = 15.0) -> None:
    anchors = [s for s in segs if s.id in embedded]
    for s in segs:
        if s.id in embedded:
            continue
        same = [a for a in anchors if a.chunk == s.chunk and a.local_speaker == s.local_speaker]
        if not same:
            continue
        near = min(same, key=lambda a: max(a.start - s.end, s.start - a.end, 0.0))
        if max(near.start - s.end, s.start - near.end, 0.0) <= reach and near.speaker != s.speaker:
            s.notes = (s.notes + f" too short to fingerprint; follows neighbour {near.speaker};").strip()
            s.speaker = near.speaker
