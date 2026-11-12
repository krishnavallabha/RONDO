"""Synthetic evidence so the decoder runs end to end before M2 exists.

Generates a made-up possession timeline and noisy contract-shaped evidence from it. The timeline
is random, NOT this match -- it never touches the ground truth.
"""

import numpy as np

from rondo import contract
from rondo.decoder import A, D, H, graded_blocks

MEAN_POSSESSION = 350   # frames, ~14 s
MEAN_STOPPAGE = 400     # frames
P_STOPPAGE_AFTER = 0.4  # chance a possession ends in a stoppage rather than a turnover


def timeline(n_frames, rng):
    """Synthetic truth: codes 0/1/2 inside the graded halves, -1 elsewhere."""
    truth = np.full(n_frames, -1, dtype=np.int8)
    for start, stop in graded_blocks():
        t, state = start, D
        while t < stop:
            mean = MEAN_STOPPAGE if state == D else MEAN_POSSESSION
            length = max(10, int(rng.exponential(mean)))
            truth[t:min(t + length, stop)] = state
            t += length
            if state == D:
                state = rng.integers(2)
            else:
                state = D if rng.random() < P_STOPPAGE_AFTER else 1 - state
    return truth


def evidence(n_frames, seed=0):
    """(evidence dict, synthetic truth codes). Seeded, so reproducible."""
    rng = np.random.default_rng(seed)
    truth = timeline(n_frames, rng)
    ev = contract.empty(n_frames)
    graded = truth >= 0
    alive = (truth == H) | (truth == A)

    ev["ball_missing"][graded] = rng.random(graded.sum()) < 0.41
    seen = graded & ~ev["ball_missing"]
    ev["ball_xy"][seen] = rng.uniform((0, 0), (105, 68), size=(seen.sum(), 2))
    for team, key in ((H, "control_H"), (A, "control_A")):
        ev[key] = (truth == team) & (rng.random(n_frames) < 0.5) | \
                  alive & (truth != team) & (rng.random(n_frames) < 0.05)
    for team, key in ((H, "touch_H"), (A, "touch_A")):
        ev[key] = (truth == team) & (rng.random(n_frames) < 0.03)
    ev["duel_zone"] = ev["control_H"] & ev["control_A"]
    noise = rng.normal(0, 0.15, n_frames)
    ev["dead_ball"][graded] = np.clip(np.where(truth == D, 0.75, 0.25) + noise, 0, 1)[graded]
    return contract.check(ev, n_frames), truth
