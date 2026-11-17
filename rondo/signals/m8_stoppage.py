"""M8 stoppage model: per-frame P(dead) from two LightGBM experts, P(home | alive) from M7's
team model retrained on more conditions. Replaces M7's p_dead / p_home in the evidence file.

    python -m rondo.signals.export --out out/evidence.npz     # adds p_dead, p_home (M8)
    python -m rondo.signals.m8_stoppage train                 # regenerate the m8_* model files

Why: on our match the ball is hidden through most long stoppages (already labelled dead), so the
stoppages M7 misses are short ones with the ball in view. M8 adds player-movement and ball-gap
features and is trained on external matches made to look like ours (a simulator fitted to our
match's label-free tracking statistics), on top of M7's five conditions.

Experts (chosen by whether the ball is seen in that frame):
  visible  M7 dead features + all stoppage features
  gap      M7 dead features that do not read the ball + stoppage features that do not read it
Features that only identify the data source (number of tracked players, standing share, ball
speed before a gap) are left out: an adversarial check separated every external match from ours
on them alone.

Simulator conditions sim0..sim2 (external labels are used only to place gaps on the EXTERNAL
match): each true stoppage is hidden (always if >= 10 s, else with p_short), the hidden span
starts mu_s s after the whistle and ends mu_e s after the restart (Normal, sd 1 s), plus
label-independent dropouts; the ball is low-passed (sigma_b frames) so fewer touches are credited,
as on our match; players get slow position noise. sim1 and sim2 jitter the parameters.
The parameters were fitted on our match's missing-ball share, gap lengths, gaps per minute and
in-gap player speed, and its credited-touch rate (no labels). Fit code and logs: out/plan30/.

Training is on external matches only; no label of our match is read anywhere.
"""

import argparse
import json
import os
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d

from rondo.signals import m2_evidence as M
from rondo.signals import m7_learned as M7

FPS = 25
H, A, D = 0, 1, 2
HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_WEIGHTS = os.path.join(HERE, "m8_team.json")
META = os.path.join(HERE, "m8_meta.json")
EXPERT_FILES = (os.path.join(HERE, "m8_dead_visible.txt"), os.path.join(HERE, "m8_dead_gap.txt"))

RELABEL = 50                    # controlled possession, as M7 ships
WEIGHTING = "sqrt"
DEAD_W = 1.5                    # class weight on dead rows; undone at apply time (log 1.5)
ROUNDS = 300
OLD_CONDS = ("full", "deadmiss", "n10dm", "n16dm", "n13corrupt")
SIM_CONDS = ("sim0", "sim1", "sim2")
CONDS = OLD_CONDS + SIM_CONDS

# Simulator fit (out/plan30/sim_params.json, scatter.json); sd_edge fixed at 1 s.
SIM_BASE = {"p_long": 1.0, "p_short": 0.2861129621474717, "mu_s": 2.776122102453391,
            "mu_e": -3.4015103096997987, "sd_edge": 1.0, "drop_rate": 1.2556399692783164,
            "drop_med": 3.3973893936258106, "drop_sig": 0.5710240440416994}
SIGMA_B = 6.141483931487858

IDENTITY = ("ntrack", "qstand_1", "qstand_75", "qstand_251", "gap_last_sp")
BALL_XS = ("still15", "still_run", "dx_line", "dy_line", "bspeed", "vis_since_gap", "vis_until_gap")
BALL_XD = ("sp", "sp_fast", "sp_still", "spm_25", "spm_75", "far", "near", "nearm_75", "edge",
           "edge_near", "still_past1.0", "still_fut1.0", "still_past4.0", "still_fut4.0",
           "fast_past1.0", "fast_fut1.0", "fast_past4.0", "fast_fut4.0", "event_past1.0",
           "event_fut1.0", "event_past4.0", "event_fut4.0", "ctrl_any_past1.0", "ctrl_any_fut1.0",
           "ctrl_any_past4.0", "ctrl_any_fut4.0")


# ---------------------------------------------------------------- helpers
def runs(flags):
    """[(start, stop)] of the True runs."""
    flags = np.asarray(flags, dtype=bool)
    e = np.diff(np.concatenate(([0], flags.view(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def run_len(mask):
    out = np.zeros(len(mask))
    for s, e in runs(mask):
        out[s:e] = e - s
    return out


def rmean(x, w):
    """Centred moving average, window w; the window shrinks at the edges."""
    x = np.asarray(x, float)
    if w <= 1:
        return x
    c = np.concatenate(([0.0], np.cumsum(x)))
    n, h = len(x), w // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def rmean_nan(x, w):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    num = rmean(np.where(ok, x, 0.0), w)
    den = rmean(ok * 1.0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def player_motion(player_xy, smooth=25):
    """Per frame: mean and 90th-percentile player speed, share walking (< 1.5 m/s), share standing
    (< 0.5 m/s), number of tracked players. Speeds per player, smoothed over `smooth` frames."""
    n, P, _ = player_xy.shape
    step = np.linalg.norm(np.diff(player_xy, axis=0), axis=2) * FPS
    step = np.vstack([step[:1], step])
    step = np.where(step > 12, np.nan, step)                            # teleports
    sp = np.empty_like(step)
    for j in range(P):
        sp[:, j] = rmean_nan(step[:, j], smooth)
    tracked = np.isfinite(player_xy[:, :, 0])
    sp = np.where(tracked, sp, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(sp, 1)
        p90 = np.nanpercentile(sp, 90, axis=1)
        cnt = np.isfinite(sp).sum(1)
        walk = np.where(cnt > 0, (sp < 1.5).sum(1) / np.maximum(cnt, 1), np.nan)
        stand = np.where(cnt > 0, (sp < 0.5).sum(1) / np.maximum(cnt, 1), np.nan)
    return {"mean": mean, "p90": p90, "walk": walk, "stand": stand,
            "ntrack": tracked.sum(1).astype(float)}


def qrank(x, ref_mask):
    """Within-match quantile of x against its values on ref_mask frames (NaN stays NaN)."""
    ref = np.sort(x[ref_mask & np.isfinite(x)])
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if len(ref):
        out[ok] = np.searchsorted(ref, x[ok], side="right") / len(ref)
    return out


# ---------------------------------------------------------------- stoppage features
def stoppage_features(ev, players, half, pitch=(105.0, 68.0)):
    """(n, 33) float32 and names. Player movement as within-match quantiles; ball-derived features
    are NaN where the ball is not seen; gap facts (length, time into, time left) are offline."""
    ball = ev["ball_xy"]
    vis = np.isfinite(ball[:, 0])
    miss = ~vis
    g = half > 0
    mot = player_motion(players)
    F = {}
    for k in ("mean", "p90", "walk", "stand"):
        q = qrank(mot[k], g)
        for w in (1, 75, 251):
            F["q{}_{}".format(k, w)] = rmean(np.nan_to_num(q, nan=0.5), w)
    qp = np.nan_to_num(qrank(mot["p90"], g), nan=0.5)
    c = np.concatenate(([0.0], np.cumsum(qp)))
    n = len(qp)
    pos = np.arange(n)
    lo, hi = np.clip(pos - 125, 0, n), np.clip(pos + 126, 0, n)
    past = (c[pos] - c[lo]) / np.maximum(pos - lo, 1)
    future = (c[hi] - c[pos + 1]) / np.maximum(hi - pos - 1, 1)
    F["p90_drop"] = past - future                                       # sustained slow-down
    F["p90_min5"] = -rmean(-np.minimum(rmean(qp, 25), 1.0), 125)
    F["p90_run_low"] = np.log1p(run_len(rmean(qp, 25) < 0.25))
    F["ntrack"] = mot["ntrack"] / max(np.median(mot["ntrack"][g]), 1)
    sp = M.ball_speed_mps(ball)
    still = vis & (np.nan_to_num(sp, nan=99) < 0.5)
    rl = run_len(still)
    F["still15"] = np.where(vis, (rl >= 37) * 1.0, np.nan)
    F["still_run"] = np.where(vis, np.log1p(rl), np.nan)
    L, W = pitch
    bx, by = ball[:, 0], ball[:, 1]
    F["dx_line"] = np.where(vis, np.clip(np.minimum(bx, L - bx), -5, 40), np.nan)
    F["dy_line"] = np.where(vis, np.clip(np.minimum(by, W - by), -5, 30), np.nan)
    F["bspeed"] = np.where(vis, np.clip(np.nan_to_num(sp, nan=0), 0, 35), np.nan)
    gl = np.zeros(n)
    into, left = np.full(n, np.nan), np.full(n, np.nan)
    last_dx, last_dy = np.full(n, np.nan), np.full(n, np.nan)
    next_dx, next_dy = np.full(n, np.nan), np.full(n, np.nan)
    last_sp = np.full(n, np.nan)
    since_gap, until_gap = np.full(n, np.nan), np.full(n, np.nan)
    gaps = runs(miss)
    for a, b in gaps:
        gl[a:b] = b - a
        into[a:b] = np.arange(b - a)
        left[a:b] = np.arange(b - a)[::-1]
        if a > 0:
            j = a - 1
            last_dx[a:b] = min(bx[j], L - bx[j])
            last_dy[a:b] = min(by[j], W - by[j])
            seg = sp[max(0, a - 12):a]
            last_sp[a:b] = np.nanmedian(seg) if np.isfinite(seg).any() else np.nan
        if b < n:
            next_dx[a:b] = min(bx[b], L - bx[b])
            next_dy[a:b] = min(by[b], W - by[b])
    ends = np.array([b for a, b in gaps]) if gaps else np.array([], int)
    starts = np.array([a for a, b in gaps]) if gaps else np.array([], int)
    vidx = np.flatnonzero(vis)
    if len(ends):
        j = np.searchsorted(ends, vidx, side="right") - 1
        since_gap[vidx] = np.where(j >= 0, vidx - ends[np.clip(j, 0, None)], 1e4)
        k = np.searchsorted(starts, vidx, side="left")
        until_gap[vidx] = np.where(k < len(starts), starts[np.clip(k, 0, len(starts) - 1)] - vidx, 1e4)
    F["gap_len"] = np.where(miss, np.log1p(gl), np.nan)
    F["gap_into"] = np.where(miss, np.log1p(into), np.nan)
    F["gap_left"] = np.where(miss, np.log1p(left), np.nan)
    F["gap_frac"] = np.where(miss, into / np.maximum(gl, 1), np.nan)
    F["gap_last_dx"] = np.clip(last_dx, -5, 40)
    F["gap_last_dy"] = np.clip(last_dy, -5, 30)
    F["gap_next_dx"] = np.clip(next_dx, -5, 40)
    F["gap_next_dy"] = np.clip(next_dy, -5, 30)
    F["gap_last_sp"] = np.clip(last_sp, 0, 35)
    F["vis_since_gap"] = np.log1p(np.minimum(since_gap, 1e4))
    F["vis_until_gap"] = np.log1p(np.minimum(until_gap, 1e4))
    gap_q = np.full(n, np.nan)
    for a, b in gaps:
        gap_q[a:b] = qp[a:b].mean()
    F["gap_mean_q"] = np.where(miss, gap_q, np.nan)
    return np.column_stack(list(F.values())).astype(np.float32), list(F)


def expert_columns(names_dead, names_stop):
    """Column indices: visible expert (XD all, XS kept), gap expert (XD kept, XS kept)."""
    vis_xs = [i for i, n in enumerate(names_stop) if n not in IDENTITY]
    gap_xd = [i for i, n in enumerate(names_dead) if n not in BALL_XD]
    gap_xs = [i for i, n in enumerate(names_stop) if n not in BALL_XS and n not in IDENTITY]
    return vis_xs, gap_xd, gap_xs


def expert_matrices(XD, XS, cols):
    vis_xs, gap_xd, gap_xs = cols
    XD, XS = XD.astype(np.float32), XS.astype(np.float32)
    return np.hstack([XD, XS[:, vis_xs]]), np.hstack([XD[:, gap_xd], XS[:, gap_xs]])


# ---------------------------------------------------------------- simulator conditions
def sim_params(i):
    p = dict(SIM_BASE)
    rng = np.random.default_rng(500 + i)
    if i > 0:                                # draws centred on the fit
        p["p_long"] = float(np.clip(p["p_long"] - abs(rng.normal(0, 0.08)), 0, 1))
        p["p_short"] = float(np.clip(p["p_short"] + rng.normal(0, 0.1), 0, 1))
        p["mu_s"] += rng.normal(0, 0.8)
        p["mu_e"] += rng.normal(0, 0.8)
        p["drop_rate"] = max(0.0, p["drop_rate"] * np.exp(rng.normal(0, 0.25)))
    p["sigma_b"] = SIGMA_B * (1 if i == 0 else float(np.exp(rng.normal(0, 0.2))))
    p["sigma_p"] = 1.3 if i == 0 else float(rng.uniform(1.0, 1.6))
    return p


def sim_mask(gt, half, p, rng):
    """Ball-missing mask for an external match: hidden stoppages with offset edges + dropouts."""
    n = len(gt)
    mask = np.zeros(n, bool)
    for h in sorted(set(half.tolist()) - {0}):
        idx = np.flatnonzero(half == h)
        s0, e0 = int(idx.min()), int(idx.max()) + 1
        for a, b in runs(gt[s0:e0] == D):
            a, b = a + s0, b + s0
            prob = p["p_long"] if b - a >= 10 * FPS else p["p_short"]
            u = rng.random()
            os_ = rng.normal(p["mu_s"], p["sd_edge"]) * FPS
            oe = rng.normal(p["mu_e"], p["sd_edge"]) * FPS
            if u >= prob:
                continue
            ga, gb = max(s0, int(round(a + os_))), min(e0, int(round(b + oe)))
            if gb - ga >= FPS // 2:
                mask[ga:gb] = True
        minutes = (e0 - s0) / FPS / 60
        k = rng.poisson(p["drop_rate"] * minutes)
        starts = rng.integers(s0, e0, size=k)
        lens = np.exp(rng.normal(np.log(p["drop_med"] * FPS), p["drop_sig"], size=k)).astype(int) + 1
        for s, l in zip(starts, lens):
            mask[s:min(e0, s + l)] = True
    return mask


def ball_smooth(ball, sigma_frames):
    """Gaussian low-pass of the ball inside each seen stretch (blunts kick impulses)."""
    if sigma_frames <= 0:
        return ball
    out = ball.copy()
    for a, b in runs(np.isfinite(ball[:, 0])):
        if b - a > 3:
            out[a:b] = gaussian_filter1d(ball[a:b], sigma_frames, axis=0, mode="nearest")
    return out


def corrupt(cond, match_id, gt, truth, meta, player_xy, real):
    """(ball, players) of one external match under one condition."""
    from rondo.signals import m2_validate as V
    if cond in OLD_CONDS:
        corruption, sigma = M7.CONDITIONS[cond]
        ball = V.CORRUPTIONS[corruption](truth, meta, player_xy, real)
        players = player_xy if sigma == 0 else player_xy + np.stack(
            [M7.slow_noise(len(ball), sigma, 10 + i) for i in range(player_xy.shape[1])], 1)
        return ball, players
    i = int(cond[3:])
    p = sim_params(i)
    seed = 7000 + 100 * i + sum(map(ord, match_id))
    mask = sim_mask(np.asarray(gt, np.int8), np.asarray(truth["half"], np.int8), p,
                    np.random.default_rng(seed))
    ball = ball_smooth(truth["ball_xy"], p["sigma_b"])
    ball[mask] = np.nan
    players = player_xy + np.stack([M7.slow_noise(len(ball), p["sigma_p"], seed + 20 + j)
                                    for j in range(player_xy.shape[1])], 1)
    return ball, players


# ---------------------------------------------------------------- training
def lgb_params(seed=0):
    return dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                lambda_l2=1.0, feature_fraction=0.8, feature_fraction_seed=seed, bagging_fraction=1.0,
                seed=seed, deterministic=True, force_row_wise=True, num_threads=8, verbose=-1)


def _match_rows(args):
    """Training rows of one external match under every condition, in CONDS order."""
    dataset, match_id, real_missing = args
    import rondo.signals  # noqa: F401
    from rondo.harness import score
    from rondo.signals import external
    gt, truth, meta = score.load(dataset, match_id)
    truth = dict(truth, codes=gt)
    player_xy, player_team = external.players(dataset, match_id)
    real = {"ev": {"ball_missing": real_missing}}
    out = []
    for cond in CONDS:
        ball, players = corrupt(cond, match_id, gt, truth, meta, player_xy, real)
        ev, _ = M.build_evidence(ball, players, player_team, tuple(meta["pitch"]))
        XT, XD, names_team, names_dead = M7.features(ev, players, player_team, truth["half"])
        XS, names_stop = stoppage_features(ev, players, truth["half"], tuple(meta["pitch"]))
        y = M7.controlled(np.asarray(gt, np.int8), RELABEL)
        xt, yt, wt, _, _ = M7.training_rows(XT, XD, y, truth["half"], WEIGHTING)
        graded = (truth["half"] > 0) & np.isin(y, (H, A, D))
        idx = np.flatnonzero(graded)[::M7.SUBSAMPLE]
        vis = ~np.asarray(ev["ball_missing"], bool)
        xv, xg = expert_matrices(XD, XS, expert_columns(names_dead, names_stop))
        iv, ig = idx[vis[idx]], idx[~vis[idx]]
        out.append((xt, yt, wt, xv[iv], y[iv] == D, xg[ig], y[ig] == D))
    return (dataset, match_id), out, (names_team, names_dead, names_stop)


def train(workers=3, log=print):
    """Refit the team model and both dead experts on every external match x condition."""
    import lightgbm as lgb
    from multiprocessing import Pool
    import rondo.signals  # noqa: F401
    from rondo.harness import score
    real_missing = np.asarray(M.main()["ball_missing"]).copy()     # label-free, as M7 uses it
    rows, names = {}, None
    with Pool(workers) as pool:
        for key, out, names in pool.imap_unordered(
                _match_rows, [(d, m, real_missing) for d, m in sorted(score.available())]):
            rows[key] = out
            log("features done: {} {}".format(*key))
    ordered = [r for key in sorted(rows) for r in rows[key]]     # fixed order -> same sums every run
    XT = np.vstack([x for r in ordered for x in (r[0].astype(np.float64), -r[0].astype(np.float64))])
    yT = np.concatenate([y for r in ordered for y in (r[1], 1 - r[1])])
    wT = np.concatenate([w for r in ordered for w in (r[2], r[2])])
    team = M7.fit_logistic(XT, yT, wT, intercept=False)
    del XT
    names_team, names_dead, names_stop = names
    with open(TEAM_WEIGHTS, "w", newline="\n") as handle:
        json.dump({"features": names_team, "intercept": False, "mu": team["mu"].tolist(),
                   "sd": team["sd"].tolist(), "beta": team["beta"].tolist()}, handle, indent=1)
        handle.write("\n")
    for k, path in enumerate(EXPERT_FILES):
        X = np.vstack([r[3 + 2 * k] for r in ordered])
        y = np.concatenate([r[4 + 2 * k] for r in ordered])
        ds = lgb.Dataset(X, label=y.astype(float), weight=np.where(y, DEAD_W, 1.0), free_raw_data=True)
        model = lgb.train(lgb_params(0), ds, num_boost_round=ROUNDS)
        model.save_model(path)
        log("expert {} trained on {} rows".format(os.path.basename(path), len(y)))
    with open(META, "w", newline="\n") as handle:
        json.dump({"dead_features": names_dead, "stoppage_features": names_stop,
                   "shift": float(np.log(DEAD_W)), "lightgbm": lgb.__version__}, handle, indent=1)
        handle.write("\n")
    return TEAM_WEIGHTS, META, EXPERT_FILES


# ---------------------------------------------------------------- apply
def probabilities(ev, player_xy, player_team, half, pitch=(105.0, 68.0)):
    """(p_dead, p_home) per frame for one match, from the committed model files."""
    import lightgbm as lgb
    with open(META) as handle:
        meta = json.load(handle)
    with open(TEAM_WEIGHTS) as handle:
        raw = json.load(handle)
    team = {"intercept": raw["intercept"], "mu": np.array(raw["mu"]), "sd": np.array(raw["sd"]),
            "beta": np.array(raw["beta"])}
    XT, XD, names_team, names_dead = M7.features(ev, player_xy, player_team, half)
    XS, names_stop = stoppage_features(ev, player_xy, half, pitch)
    if (names_team != raw["features"] or names_dead != meta["dead_features"]
            or names_stop != meta["stoppage_features"]):
        raise ValueError("M8 model files were trained on a different feature list; retrain")
    xv, xg = expert_matrices(XD, XS, expert_columns(names_dead, names_stop))
    vis = ~np.asarray(ev["ball_missing"], bool)
    z = np.empty(len(vis))
    for k, (rows, X) in enumerate(((vis, xv), (~vis, xg))):
        if rows.any():
            z[rows] = lgb.Booster(model_file=EXPERT_FILES[k]).predict(X[rows], raw_score=True)
    z -= meta["shift"]                          # undo the class weight: back to the training prior
    p_dead = 1 / (1 + np.exp(-z))
    p_home = 1 / (1 + np.exp(-M7.logit(team, XT.astype(np.float64))))
    return p_dead, p_home


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    print("wrote", train(args.workers))


if __name__ == "__main__":
    main()
