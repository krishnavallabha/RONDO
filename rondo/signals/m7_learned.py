"""M7 learned emissions: per-frame P(dead) and P(home | alive) from two logistic models.

The models are trained on external matches only (rondo.harness: IDSSE + Metrica, public labels),
each seen under five ball / player-noise conditions, and applied to our match unchanged. Their
weights are committed in m7_weights.json, so the submission does not need the external data:

    python -m rondo.signals.export --out out/evidence.npz     # adds p_dead, p_home
    python -m rondo.signals.m7_learned train                  # regenerate m7_weights.json

No label of our match is read anywhere. Training uses our match in two label-free ways: its
missing-ball mask (the "corrupt" condition transplants it onto external matches, as
m2_validate does) and nothing else; at apply time player speed is divided by the match's own
median, which is a unit normalisation.

Features (all per frame, 25 fps):
  team model  soft proximity of the ball to each team at several windows, sustained control,
              credited and soft touches as past/future decays, team width. Every feature is
              "home minus away", so swapping the teams flips P(home) exactly (no intercept).
  dead model  ball missing and for how long, player speed relative to the match median, the
              dead_ball evidence levels, ball still / out / fast, kicks, distance to the lines.
"""

import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from rondo.signals import m2_evidence as M

FPS = 25
H, A, D = 0, 1, 2
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m7_weights.json")
# name: (ball corruption in m2_validate.CORRUPTIONS, slow player-position noise sigma in metres)
CONDITIONS = {"full": ("full", 0.0), "deadmiss": ("deadmiss", 0.0), "n10dm": ("deadmiss", 1.0),
              "n16dm": ("deadmiss", 1.6), "n13corrupt": ("corrupt", 1.3)}
SUBSAMPLE = 4           # every 4th graded frame is a training row
L2 = 1e-3


# ---------------------------------------------------------------- helpers
def rmean(x, w):
    """Centred moving average, window w (odd); the window shrinks at the edges."""
    if w <= 1:
        return x
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def run_len(mask):
    """Length of the True run each frame belongs to (0 where False)."""
    out = np.zeros(len(mask))
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[s:e] = e - s
    return out


def since(mask):
    """Frames since the last True and until the next True (1e4 when there is none)."""
    n = len(mask)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return np.full(n, 1e4), np.full(n, 1e4)
    pos = np.arange(n)
    j = np.searchsorted(idx, pos, side="right")
    prev = np.where(j > 0, pos - idx[np.clip(j - 1, 0, len(idx) - 1)], 1e4)
    nxt = np.where(j < len(idx), idx[np.clip(j, 0, len(idx) - 1)] - pos, 1e4)
    return prev, nxt


def decay(x, tau_s, forward=True):
    """Sum of past (forward) or future impulses of x, each decaying with time constant tau_s."""
    a = np.exp(-1.0 / (tau_s * FPS))
    if forward:
        return lfilter([1.0], [1.0, -a], x)
    return lfilter([1.0], [1.0, -a], x[::-1])[::-1]


def slow_noise(n, sigma, seed, tscale=100):
    """(n, 2) position noise that drifts over ~tscale frames, so player speeds stay realistic."""
    z = np.random.default_rng(seed).standard_normal((n, 2))
    s = gaussian_filter1d(z, tscale, axis=0, mode="wrap")
    return s / s.std(0) * sigma


def kick_events(ball):
    """Frames where the ball is kicked (impulse or sharp turn), before any player is credited."""
    vb, va = M.ball_velocity(ball)
    vbs, vas = M.ball_velocity(ball, M.VEL_WINDOW_SHORT)
    vb = np.where(np.isnan(vb), vbs, vb)
    va = np.where(np.isnan(va), vas, va)
    imp = np.linalg.norm(va - vb, axis=1)
    sb, sa = np.linalg.norm(vb, axis=1), np.linalg.norm(va, axis=1)
    turn = M.angle_diff(np.arctan2(va[:, 1], va[:, 0]), np.arctan2(vb[:, 1], vb[:, 0]))
    event = (imp > M.TOUCH_IMPULSE) | ((turn > M.TOUCH_DIRECTION_THRESHOLD)
                                       & (np.minimum(sb, sa) > M.TOUCH_DIRECTION_MIN_SPEED))
    impn = np.where(np.isnan(imp), -np.inf, imp)
    pad = np.pad(impn, M.TOUCH_NMS, constant_values=-np.inf)
    return event & (impn >= np.lib.stride_tricks.sliding_window_view(pad, 2 * M.TOUCH_NMS + 1).max(1))


# ---------------------------------------------------------------- features
def features(ev, player_xy, player_team, half):
    """Team and dead feature matrices for one match.

    ev           evidence from m2_evidence.build_evidence (its cleaned ball_xy is used)
    player_xy    (n, P, 2) metres, player_team (P,) 0 home / 1 away
    half         (n,) 0 outside the graded halves; only used for the median player speed
    Returns XT (n, 50), XD (n, 54), both float32, and their column names.
    """
    from rondo.signals.m2_validate import player_speed
    ball = ev["ball_xy"]
    nd = M.team_nearest(ball, player_xy, player_team)
    speed = M.ball_speed_mps(ball)
    vis = np.isfinite(nd).all(1) & np.isfinite(ball[:, 0])
    dH = np.where(vis, nd[:, 0], 50.0)
    dA = np.where(vis, nd[:, 1], 50.0)
    g = dA - dH
    dmin = np.minimum(dH, dA)
    v = vis.astype(float)
    sp = np.where(np.isfinite(speed), speed, 0.0)
    ps_raw = player_speed(player_xy)
    med = np.nanmedian(ps_raw[half > 0])
    ps = np.where(np.isfinite(ps_raw), ps_raw, med) / med          # ratio to the match median

    T = {}
    b1 = np.tanh(g / 1.0) * np.exp(-dmin / 2.0) * v
    b2 = np.tanh(g / 3.0) * np.exp(-dmin / 5.0) * v
    b3 = np.tanh(g / 8.0) * v
    for name, b in (("b1", b1), ("b2", b2), ("b3", b3)):
        for w in (1, 13, 37, 75, 151, 301):
            T["{}_{}".format(name, w)] = rmean(b, w)
    slow = (sp < 10) * 1.0
    T["b1slow_37"] = rmean(b1 * slow, 37)
    T["b1slow_151"] = rmean(b1 * slow, 151)
    ctrl = ev["control_H"].astype(float) - ev["control_A"].astype(float)
    tch = ev["touch_H"].astype(float) - ev["touch_A"].astype(float)
    for w in (1, 25, 75, 151):
        T["ctrl_{}".format(w)] = rmean(ctrl, w)
    event = kick_events(ball).astype(float)
    stch = event * np.tanh(g / 2.0) * np.exp(-dmin / 4.0) * v      # soft touch: any kick, credited softly
    for name, x in (("ctrl", ctrl), ("tch", tch), ("stch", stch), ("b2", b2)):
        for tau in (1.0, 3.0, 8.0):
            scale = 1.0 if name in ("tch", "stch") else 1.0 / (tau * FPS)
            T["{}_past{}".format(name, tau)] = decay(x, tau, True) * scale
            T["{}_fut{}".format(name, tau)] = decay(x, tau, False) * scale
    shape = np.nan_to_num(np.asarray(ev["shape_spread"], float))
    for w in (75, 301):
        T["shape_{}".format(w)] = rmean(shape, w)

    Dm = {}
    miss = 1.0 - v
    rl = run_len(miss > 0)
    prev, nxt = since(vis)
    db = np.asarray(ev["dead_ball"], float)
    Dm["miss"] = miss
    Dm["missrun"] = np.log1p(rl) * miss
    Dm["miss_long"] = (rl > 125) * 1.0
    Dm["miss_vlong"] = (rl > 375) * 1.0
    Dm["prev"] = np.log1p(np.minimum(prev, 2000))
    Dm["nxt"] = np.log1p(np.minimum(nxt, 2000))
    Dm["gap_pos"] = np.where(miss > 0, np.minimum(prev, nxt) / np.maximum(rl, 1), 0.0)
    for w in (1, 25, 75, 151, 301):
        Dm["ps_{}".format(w)] = rmean(ps, w)
    Dm["ps_slow"] = (rmean(ps, 25) < 0.6) * 1.0
    Dm["ps_slow_miss"] = Dm["ps_slow"] * miss
    Dm["ps_miss"] = rmean(ps, 75) * miss
    Dm["ps_min151"] = -rmean(-np.minimum(rmean(ps, 25), 1.5), 151)
    for level in (0.3, 0.7, 0.8, 0.9, 1.0):
        Dm["db_{}".format(level)] = (np.abs(db - level) < 0.05) * 1.0
    Dm["db_s75"] = rmean(db, 75)
    oob = (db >= 0.89) * 1.0
    still = (sp < 0.3) * v
    control_any = (ev["control_H"] | ev["control_A"]) * 1.0
    for name, x in (("oob", oob), ("still", still), ("ctrl_any", control_any), ("event", event),
                    ("fast", (sp > 3) * v)):
        for tau in (1.0, 4.0):
            Dm["{}_past{}".format(name, tau)] = decay(x, tau, True) / (tau * FPS)
            Dm["{}_fut{}".format(name, tau)] = decay(x, tau, False) / (tau * FPS)
    Dm["sp"] = np.minimum(sp, 30) * v
    Dm["sp_fast"] = (sp > 3) * v
    Dm["sp_still"] = still
    for w in (25, 75):
        Dm["spm_{}".format(w)] = rmean(np.minimum(sp, 30) * v, w)
    Dm["far"] = np.minimum(dmin, 20) * v
    Dm["near"] = (dmin < 1.5) * v
    Dm["nearm_75"] = rmean((dmin < 2.0) * v, 75)
    # Distance to the nearest line on a nominal 105 x 68 pitch (every match here is within 0.3 m of it).
    bx = np.where(vis, ball[:, 0], 52.5)
    by = np.where(vis, ball[:, 1], 34.0)
    edge = np.minimum(np.minimum(bx, 105 - bx), np.minimum(by, 68 - by))
    Dm["edge"] = np.clip(edge, -5, 20) * v
    Dm["edge_near"] = (edge < 2.0) * v
    Dm["vis_151"] = rmean(v, 151)
    Dm["vis_751"] = rmean(v, 751)
    return (np.column_stack(list(T.values())).astype(np.float32),
            np.column_stack(list(Dm.values())).astype(np.float32), list(T), list(Dm))


# ---------------------------------------------------------------- model
def fit_logistic(X, y, w=None, l2=L2, iters=60, intercept=True):
    """Weighted, L2-regularised logistic regression by Newton steps on standardised features."""
    w = np.ones(len(X)) if w is None else w
    mu = X.mean(0) if intercept else np.zeros(X.shape[1])
    sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    if intercept:
        Z = np.column_stack([np.ones(len(X)), Z])
    beta = np.zeros(Z.shape[1])
    reg = l2 * w.sum() * np.eye(Z.shape[1])
    if intercept:
        reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ beta)))
        step = np.linalg.solve((Z * (w * p * (1 - p))[:, None]).T @ Z + reg, Z.T @ (w * (y - p)) - reg @ beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return {"mu": mu, "sd": sd, "beta": beta, "intercept": intercept}


def logit(model, X):
    Z = (np.asarray(X, dtype=np.float64) - model["mu"]) / model["sd"]
    if model["intercept"]:
        Z = np.column_stack([np.ones(len(Z)), Z])
    return Z @ model["beta"]


WEIGHTINGS = {"clip": 1.0, "sqrt": 0.5, "none": 0.0}   # exponent on 200 / run length


def spell_weights(gt, weighting="clip"):
    """Per-frame weights: a run of n frames weighs clip((200 / n) ** p, 0.25, 4), p by weighting.
    clip (p=1) is possession-equal; sqrt is milder; none weighs every frame the same."""
    w = np.ones(len(gt))
    p = WEIGHTINGS[weighting]
    change = np.flatnonzero(np.diff(gt) != 0) + 1
    for s, e in zip(np.concatenate(([0], change)), np.concatenate((change, [len(gt)]))):
        w[s:e] = np.clip((200.0 / (e - s)) ** p, 0.25, 4.0)
    return w


CONTEXT_WINDOWS = (1, 25, 75, 251, 501)     # frames, centred
CONTEXT_SIDE = 250                          # 10 s looked at only before, or only after


def _side_means(x, w):
    """Mean of x over the w frames before each frame, and over the w frames after it."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    n, pos = len(x), np.arange(len(x))
    lo, hi = np.clip(pos - w, 0, n), np.clip(pos + 1 + w, 0, n)
    past = (c[pos] - c[lo]) / np.maximum(pos - lo, 1)
    future = (c[hi] - c[pos + 1]) / np.maximum(hi - pos - 1, 1)
    return past, future


def context_features(p_dead, p_home):
    """Second stage: the first stage's own logits over windows up to +-10 s.
    Team features are odd in the home logit, so swapping teams still flips P(home) exactly."""
    eps = 1e-6
    ld = np.log(np.clip(p_dead, eps, 1 - eps) / np.clip(1 - p_dead, eps, 1))
    lh = np.log(np.clip(p_home, eps, 1 - eps) / np.clip(1 - p_home, eps, 1))
    alive_lh = lh * (1 - p_dead)
    xd = [rmean(ld, w) for w in CONTEXT_WINDOWS] + list(_side_means(ld, CONTEXT_SIDE)) + [
        rmean(np.abs(lh), w) for w in (25, 251)]
    xt = [rmean(lh, w) for w in CONTEXT_WINDOWS] + list(_side_means(lh, CONTEXT_SIDE)) + [
        rmean(alive_lh, w) for w in (75, 251)]
    return np.stack(xt, 1), np.stack(xd, 1)


def controlled(gt, min_frames):
    """External labels under CHALLENGE.md's "controlled possession": a team's alive run shorter
    than min_frames, straight after the other team's alive run, is a touch without control, so
    possession stays with the last team. Runs after a stoppage are kept (a restart has a clear owner).
    """
    gt = np.array(gt, copy=True)
    if min_frames <= 1:
        return gt
    change = np.flatnonzero(np.diff(gt) != 0) + 1
    starts, ends = np.concatenate(([0], change)), np.concatenate((change, [len(gt)]))
    for s, e in zip(starts[1:], ends[1:]):
        prev = gt[s - 1]          # already relabelled, so a chain of short touches all merges
        if gt[s] in (H, A) and prev in (H, A) and prev != gt[s] and e - s < min_frames:
            gt[s:e] = prev
    return gt


def training_rows(XT, XD, gt, half, weighting="clip"):
    """Rows of one match-condition: (team X, team y, team weight, dead X, dead y)."""
    graded = (half > 0) & np.isin(gt, (H, A, D))
    idx = np.flatnonzero(graded)[::SUBSAMPLE]
    alive = idx[gt[idx] != D]
    return (XT[alive], (gt[alive] == H) * 1.0, spell_weights(gt, weighting)[alive], XD[idx],
            (gt[idx] == D) * 1.0)


def fit(rows):
    """Both models from a list of training_rows() outputs."""
    XT, yT, wT, XD, yD = [], [], [], [], []
    for xt, yt, wt, xd, yd in rows:
        xt = xt.astype(np.float64)
        XT += [xt, -xt]            # both viewpoints: every team feature is home minus away
        yT += [yt, 1 - yt]
        wT += [wt, wt]
        XD.append(xd.astype(np.float64))
        yD.append(yd)
    team = fit_logistic(np.vstack(XT), np.concatenate(yT), np.concatenate(wT), intercept=False)
    dead = fit_logistic(np.vstack(XD), np.concatenate(yD))
    return team, dead


def save_weights(team, dead, names_team, names_dead, path=WEIGHTS):
    def pack(model, names):
        return {"features": names, "intercept": bool(model["intercept"]),
                "mu": model["mu"].tolist(), "sd": model["sd"].tolist(), "beta": model["beta"].tolist()}
    with open(path, "w", newline="\n") as handle:
        json.dump({"team": pack(team, names_team), "dead": pack(dead, names_dead)}, handle, indent=1)
        handle.write("\n")
    return path


def load_weights(path=WEIGHTS):
    with open(path) as handle:
        raw = json.load(handle)
    out = {}
    for key in ("team", "dead"):
        m = raw[key]
        out[key] = {"features": m["features"], "intercept": m["intercept"], "mu": np.array(m["mu"]),
                    "sd": np.array(m["sd"]), "beta": np.array(m["beta"])}
    return out


def probabilities(ev, player_xy, player_team, half, weights=None):
    """(p_dead, p_home) per frame for one match, from the committed weights."""
    weights = load_weights() if weights is None else weights
    XT, XD, names_team, names_dead = features(ev, player_xy, player_team, half)
    if names_team != weights["team"]["features"] or names_dead != weights["dead"]["features"]:
        raise ValueError("m7_weights.json was trained on a different feature list; retrain")
    p_dead = 1 / (1 + np.exp(-logit(weights["dead"], XD)))
    p_home = 1 / (1 + np.exp(-logit(weights["team"], XT)))
    return p_dead, p_home


# ---------------------------------------------------------------- training on the external matches
def _match_rows(args):
    dataset, match_id, real_missing, relabel, weighting = args
    import rondo.signals  # noqa: F401
    from rondo.harness import score
    from rondo.signals import external
    from rondo.signals import m2_validate as V
    real = {"ev": {"ball_missing": real_missing}}
    gt, truth, meta = score.load(dataset, match_id)
    truth = dict(truth, codes=gt)
    player_xy, player_team = external.players(dataset, match_id)
    out = {}
    for cond, (corruption, sigma) in CONDITIONS.items():
        ball = V.CORRUPTIONS[corruption](truth, meta, player_xy, real)
        players = player_xy if sigma == 0 else player_xy + np.stack(
            [slow_noise(len(ball), sigma, 10 + i) for i in range(player_xy.shape[1])], 1)
        ev, _ = M.build_evidence(ball, players, player_team, tuple(meta["pitch"]))
        XT, XD, names_team, names_dead = features(ev, players, player_team, truth["half"])
        out["{}_{}_{}".format(dataset, match_id, cond)] = training_rows(
            XT, XD, controlled(gt, relabel), truth["half"], weighting)
    return out, names_team, names_dead


def train(workers=3, path=WEIGHTS, log=print, relabel=0, weighting="clip"):
    """Refit both models on every external match x condition and rewrite the weights file.
    relabel > 0: train on controlled-possession labels (see controlled)."""
    from multiprocessing import Pool
    import rondo.signals  # noqa: F401
    from rondo.harness import score
    real_missing = np.asarray(M.main()["ball_missing"]).copy()     # label-free: where our tracker lost the ball
    rows = {}
    with Pool(workers) as pool:
        for out, names_team, names_dead in pool.imap_unordered(
                _match_rows, [(d, m, real_missing, relabel, weighting) for d, m in score.available()]):
            rows.update(out)
            log("features done: {} match-conditions".format(len(rows)))
    team, dead = fit([rows[name] for name in sorted(rows)])        # fixed order -> same sums every run
    log("trained on {} match-conditions; {} team features, {} dead features".format(
        len(rows), len(names_team), len(names_dead)))
    return save_weights(team, dead, names_team, names_dead, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=WEIGHTS)
    parser.add_argument("--relabel", type=int, default=0,
                        help="train on controlled-possession labels: min frames of a real change")
    parser.add_argument("--weighting", choices=sorted(WEIGHTINGS), default="clip",
                        help="per-frame training weight by run length (see spell_weights)")
    args = parser.parse_args(argv)
    print("wrote", train(args.workers, args.out, relabel=args.relabel, weighting=args.weighting))


if __name__ == "__main__":
    main()
