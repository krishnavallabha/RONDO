"""M3 -- three-state HMM decoder {H, A, D} over the emission contract.

Nov 12 skeleton: placeholder emissions and transition costs, offline Viterbi run separately in
each graded half. Minimum durations, restart logic, boundary snapping and expected-F1 decoding
build on top of this (Nov 13-14).
"""

import numpy as np

from games import GAME, half_video_blocks

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2
EPS = 1e-6

# Per-frame transition probabilities at 25 fps. Placeholders until C calibrates them on external
# matches. Direct H<->A (interception) is allowed but rarer than going through D, otherwise the
# decoder routes every hand-over through a spurious stoppage.
DEFAULTS = {
    "p_switch_direct": 2e-4,  # H -> A or A -> H
    "p_to_dead": 1e-3,        # H/A -> D
    "p_from_dead": 2e-3,      # D -> H or D -> A (each)
    "control_weight": 0.4,    # how far sustained control pulls team share from 0.5
    "touch_weight": 0.05,
}


def graded_blocks(game=GAME):
    """[(start, stop)] half-open video-frame spans that are scored -- the two halves."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def emission_loglik(ev, params=DEFAULTS):
    """(n, 3) log-likelihood of each state per frame. Placeholder model, not calibrated."""
    p_dead = np.clip(ev["dead_ball"], EPS, 1 - EPS)
    share = (0.5
             + params["control_weight"] * (ev["control_H"].astype(float) - ev["control_A"])
             + params["touch_weight"] * (ev["touch_H"].astype(float) - ev["touch_A"]))
    share = np.clip(share, EPS, 1 - EPS)
    return np.log(np.stack([(1 - p_dead) * share, (1 - p_dead) * (1 - share), p_dead], axis=1))


def log_transitions(params=DEFAULTS):
    """(3, 3) log P(next = j | now = i)."""
    sw, to_d, from_d = params["p_switch_direct"], params["p_to_dead"], params["p_from_dead"]
    trans = np.array([
        [1 - sw - to_d, sw, to_d],
        [sw, 1 - sw - to_d, to_d],
        [from_d, from_d, 1 - 2 * from_d],
    ])
    return np.log(trans)


def viterbi(loglik, log_trans, log_init):
    """Most likely state path. Ties go to the lower state index, so output is deterministic."""
    n, k = loglik.shape
    back = np.empty((n, k), dtype=np.int8)
    score = log_init + loglik[0]
    for t in range(1, n):
        cand = score[:, None] + log_trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + loglik[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(ev, n_frames, params=DEFAULTS, game=GAME):
    """Labels for every video frame: H/A/D inside the graded halves, X everywhere else.

    X policy: abstain only where nothing is scored. Inside a half an X costs macroF1 and gains
    nothing on spellF1 (the scorer forward-fills it), so the decoder always commits there.
    Each half is decoded on its own and starts in D (kickoff), so no state crosses halftime.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    loglik = emission_loglik(ev, params)
    log_trans = log_transitions(params)
    log_init = np.log([EPS, EPS, 1.0])
    for start, stop in graded_blocks(game):
        path = viterbi(loglik[start:stop], log_trans, log_init)
        labels[start:stop] = np.array(STATES)[path]
    return labels
