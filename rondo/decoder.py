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
    "min_alive": 75,          # frames; 3 s. Chosen on the external harness (m3_eval, Nov 14)
    "min_dead": 25,           # frames; minimum D spell
    "m7_prior_dead": 0.42,    # dead share of M7's external training matches (EXTERNAL_DATA.md)
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
    if "p_dead" in ev and "p_home" in ev:
        # M7: a classifier gives posteriors P(state | x). The HMM wants likelihoods, and its
        # transitions already carry the class prior, so divide it out (hybrid HMM/NN).
        p_dead = np.clip(ev["p_dead"], EPS, 1 - EPS)
        share = np.clip(ev["p_home"] + params["restart_weight"] * pull, EPS, 1 - EPS)
        q = params["m7_prior_dead"]
        post = np.stack([(1 - p_dead) * share, (1 - p_dead) * (1 - share), p_dead], axis=1)
        return np.log(post) - np.log([(1 - q) / 2, (1 - q) / 2, q])
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
            others = [t for t in range(3) if t != s]
            trans[last[s], first[others]] = base[s, others]  # no re-entry into its own chain
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


def chain_viterbi(loglik, params=DEFAULTS):
    """viterbi() over expand(params), O(sub-states) per frame instead of O(sub-states^2).

    Inside a chain the only move is a shift by one, so a step is: shift every sub-state forward,
    let each chain's first sub-state take the best entry from another chain's last, and let each
    last sub-state keep itself if staying beats arriving. Same path as viterbi() on the dense
    matrix (test_decoder checks it); needed once min_alive is long (75 -> 175 sub-states).
    """
    state_of, _, log_init = expand(params)
    lengths = np.bincount(state_of, minlength=3)
    first = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    last = first + lengths - 1
    base = log_transitions(params)
    off = base.copy()
    np.fill_diagonal(off, -np.inf)  # entering a chain always comes from another state
    stay = np.diag(base)
    n, k = len(loglik), len(state_of)
    emit = loglik[:, state_of]
    shift_src = np.arange(-1, k - 1, dtype=np.int16)
    back = np.empty((n, k), dtype=np.int16)
    score = log_init + emit[0]
    for t in range(1, n):
        new = np.empty(k)
        new[1:] = score[:-1]
        src = shift_src.copy()
        entry = score[last][:, None] + off
        best = entry.argmax(axis=0)
        new[first] = entry[best, np.arange(3)]
        src[first] = last[best]
        kept = score[last] + stay
        better = kept > new[last]
        new[last] = np.where(better, kept, new[last])
        src[last] = np.where(better, last, src[last])
        back[t] = src
        score = new + emit[t]
    path = np.empty(n, dtype=np.int16)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return state_of[path].astype(np.int8)


def _chain_parts(params):
    state_of, _, log_init = expand(params)
    lengths = np.bincount(state_of, minlength=3)
    first = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    last = first + lengths - 1
    return state_of, log_init, lengths, first, last, log_transitions(params)


def chain_forward(loglik, params=DEFAULTS):
    """(n, K) log forward probabilities over the duration chains, each row max-normalised."""
    state_of, log_init, lengths, first, last, base = _chain_parts(params)
    off = base.copy()
    np.fill_diagonal(off, -np.inf)
    stay = np.diag(base)
    emit = loglik[:, state_of]
    alpha = np.empty(emit.shape, dtype=np.float32)
    a = log_init + emit[0]
    alpha[0] = a - a.max()
    for t in range(1, len(emit)):
        a = alpha[t - 1].astype(float)
        new = np.empty_like(a)
        new[1:] = a[:-1]
        new[first] = np.logaddexp.reduce(a[last][:, None] + off, axis=0)
        new[last] = np.logaddexp(new[last], a[last] + stay)
        new += emit[t]
        alpha[t] = new - new.max()
    return alpha


def sample_paths(loglik, params=DEFAULTS, n_samples=40, seed=0):
    """(n_samples, n) state paths drawn from the HMM posterior: forward filter, backward sample.

    Each sub-state has at most three predecessors (previous sub-state, itself, or another chain's
    last), so a backward step is a 3-way choice per sample, done for all samples at once with the
    Gumbel-max trick. Seeded, so the draws -- and the submission -- are reproducible.
    """
    state_of, _, lengths, first, last, base = _chain_parts(params)
    alpha = chain_forward(loglik, params)
    k = len(state_of)
    preds = np.zeros((k, 3), dtype=np.int64)
    logw = np.full((k, 3), -np.inf)
    for s in range(3):
        others = [t for t in range(3) if t != s]
        for j in range(first[s], last[s] + 1):
            if j == first[s]:
                preds[j, :2] = last[others]
                logw[j, :2] = base[others, s]
            else:
                preds[j, 0], logw[j, 0] = j - 1, 0.0
            if j == last[s]:
                preds[j, 2], logw[j, 2] = j, base[s, s]
    rng = np.random.default_rng(seed)
    n = len(loglik)
    z = np.empty((n_samples, n), dtype=np.int16)
    p = np.exp(alpha[-1].astype(float))
    z[:, -1] = rng.choice(k, size=n_samples, p=p / p.sum())
    for t in range(n - 1, 0, -1):
        cand = preds[z[:, t]]
        w = alpha[t - 1][cand] + logw[z[:, t]] + rng.gumbel(size=cand.shape)
        z[:, t - 1] = cand[np.arange(n_samples), w.argmax(axis=1)]
    return state_of[z].astype(np.int8)


def mbr_half(loglik, params=DEFAULTS, n_samples=40, seed=0):
    """Expected-spellF1 decoding: of the Viterbi path and the posterior samples, the one with the
    highest mean spellF1 against the samples, scored by the official possession_metrics code.

    shortcut: 40 samples, not the plan's 300 (pairwise scoring is O(samples^2) at ~15 ms a pair);
    on the harness 40 and 100 samples score the same (docs/decoder.md), so more would not help.
    """
    import possession_metrics as pm
    samples = sample_paths(loglik, params, n_samples, seed)
    candidates = np.vstack([chain_viterbi(loglik, params)[None], samples])
    runs = [(0, len(loglik))]
    expected = [np.mean([pm.spell_scores(ref, cand, runs)["mean_f1"] for ref in samples])
                for cand in candidates]
    return candidates[int(np.argmax(expected))]


def snap_switches(codes, touch_H, touch_A, window):
    """Move each direct H<->A switch to the new owner's first touch within +-window frames.

    Possession changes when the other team touches the ball; the HMM places the switch where the
    evidence balance tips, which is spread ~+-0.7 s around it on the harness. Switches through D,
    and switches with no touch nearby, are left alone; a switch never crosses its neighbours.
    """
    codes = codes.copy()
    touches = {H: np.asarray(touch_H, bool), A: np.asarray(touch_A, bool)}
    edges = np.flatnonzero(np.diff(codes) != 0) + 1
    for i, j in enumerate(edges):
        old, new = int(codes[j - 1]), int(codes[j])
        if D in (old, new):
            continue
        lo = max(j - window, edges[i - 1] + 1 if i else 1)
        hi = min(j + window, edges[i + 1] - 1 if i + 1 < len(edges) else len(codes) - 1)
        hits = np.flatnonzero(touches[new][lo:hi + 1])
        if len(hits):
            k = lo + int(hits[0])
            codes[min(j, k):max(j, k)] = new if k < j else old
    return codes


def decode_half(loglik, params=DEFAULTS):
    """Codes 0/1/2 (H/A/D) for one half, from its (n, 3) emission log-likelihoods."""
    return chain_viterbi(loglik, params)


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
