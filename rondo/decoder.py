"""M3 -- three-state HMM decoder {H, A, D} over the emission contract.

Offline Viterbi, run separately in each graded half. Minimum durations come from expanding each
state into a chain of sub-states (an explicit-duration HMM): a spell must walk the whole chain
before it may leave, so H/A spells last at least `min_alive` frames and D at least `min_dead`.
Restarts: where M2 located a restart and its taker, the team share is pulled to that team.
Boundary snapping and expected-F1 decoding build on top of this (Nov 14).
"""

import numpy as np

from games import GAME, half_video_blocks

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2
EPS = 1e-6

# Per-frame transition probabilities at 25 fps. Direct H<->A (interception) is allowed but rarer
# than going through D, otherwise the decoder routes every hand-over through a spurious stoppage.
# p_to_dead and touch_weight: A's decoder_fix, chosen on the 2 Metrica matches and confirmed on
# IDSSE (rondo/signals/m2_validate.py). The rest are placeholders until C's Nov 14 sweeps.
DEFAULTS = {
    "p_switch_direct": 2e-4,  # H -> A or A -> H
    "p_to_dead": 4e-3,        # H/A -> D
    "p_from_dead": 2e-3,      # D -> H or D -> A (each)
    "control_weight": 0.4,    # how far sustained control pulls team share from 0.5
    "touch_weight": 0.2,
    "restart_weight": 0.45,   # how far a located restart pulls team share toward its taker
    "min_alive": 10,          # frames; minimum H or A spell
    "min_dead": 25,           # frames; minimum D spell
}


def graded_blocks(game=GAME):
    """[(start, stop)] half-open video-frame spans that are scored -- the two halves."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def emission_loglik(ev, params=DEFAULTS):
    """(n, 3) log-likelihood of each state per frame.

    dead_ball is a heuristic score where 0.5 means "no idea". Read raw, 0.5 against 0.25 for each
    team would make every evidence-free frame lean dead, so it is remapped to p / (p + 2(1 - p)):
    0.5 -> 1/3 (all three states equal), 0 and 1 fixed, order kept.
    """
    raw = np.asarray(ev["dead_ball"], dtype=float)
    p_dead = np.clip(raw / (raw + 2.0 * (1.0 - raw)), EPS, 1 - EPS)
    restart = np.asarray(ev["restart_team"], dtype=float)
    if not np.isin(restart[~np.isnan(restart)], (0, 1)).all():
        # M2's internal -1 once leaked here and pulled every frame to H; see export.to_contract.
        raise ValueError("restart_team must be 0, 1 or NaN (contract), not M2's internal -1")
    pull = np.where(np.isnan(restart), 0.0, 1.0 - 2.0 * np.nan_to_num(restart))  # +1 H, -1 A
    share = (0.5
             + params["control_weight"] * (ev["control_H"].astype(float) - ev["control_A"])
             + params["touch_weight"] * (ev["touch_H"].astype(float) - ev["touch_A"])
             + params["restart_weight"] * pull)
    share = np.clip(share, EPS, 1 - EPS)
    return np.log(np.stack([(1 - p_dead) * share, (1 - p_dead) * (1 - share), p_dead], axis=1))


def log_transitions(params=DEFAULTS):
    """(3, 3) log P(next = j | now = i), once a spell has reached its minimum duration."""
    sw, to_d, from_d = params["p_switch_direct"], params["p_to_dead"], params["p_from_dead"]
    trans = np.array([
        [1 - sw - to_d, sw, to_d],
        [sw, 1 - sw - to_d, to_d],
        [from_d, from_d, 1 - 2 * from_d],
    ])
    return np.log(trans)


def expand(params=DEFAULTS):
    """(state_of, log_trans, log_init) over the duration chains.

    State s owns sub-states 0..m_s-1. Entering s lands on sub-state 0; sub-state i moves to i+1
    with certainty; only the last one may stay or leave, with the 3x3 probabilities. A minimum
    of 1 gives back the plain HMM. A half may open in any state (first sub-state) -- the first
    half is graded from the kickoff itself, the second from well before it.
    """
    lengths = (params["min_alive"], params["min_alive"], params["min_dead"])
    state_of = np.repeat(np.arange(3), lengths)
    first = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    last = first + np.array(lengths) - 1
    k = len(state_of)
    with np.errstate(divide="ignore"):
        trans = np.full((k, k), -np.inf)
        base = log_transitions(params)
        for s in range(3):
            for i in range(first[s], last[s]):
                trans[i, i + 1] = 0.0
            trans[last[s], first] = base[s]
            trans[last[s], last[s]] = base[s, s]
        init = np.full(k, -np.inf)
        init[first] = np.log(1.0 / 3)
    return state_of, trans, init


def viterbi(loglik, log_trans, log_init):
    """Most likely state path. Ties go to the lower state index, so output is deterministic."""
    n, k = loglik.shape
    back = np.empty((n, k), dtype=np.int16)
    score = log_init + loglik[0]
    for t in range(1, n):
        cand = score[:, None] + log_trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + loglik[t]
    path = np.empty(n, dtype=np.int16)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_half(loglik, params=DEFAULTS):
    """Codes 0/1/2 (H/A/D) for one half, from its (n, 3) emission log-likelihoods."""
    state_of, log_trans, log_init = expand(params)
    return state_of[viterbi(loglik[:, state_of], log_trans, log_init)].astype(np.int8)


def decode(ev, n_frames, params=DEFAULTS, game=GAME):
    """Labels for every video frame: H/A/D inside the graded halves, X everywhere else.

    X policy: abstain only where nothing is scored. Inside a half an X costs macroF1 and gains
    nothing on spellF1 (the scorer forward-fills it), so the decoder always commits there.
    Each half is decoded on its own, so no state crosses halftime.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    loglik = emission_loglik(ev, params)
    for start, stop in graded_blocks(game):
        labels[start:stop] = np.array(STATES)[decode_half(loglik[start:stop], params)]
    return labels
