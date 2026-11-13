"""Run our pipeline on the harness's external matches, with and without ball-free features.

    python -m rondo.signals.m2_validate

For every external match: players + ball -> build_evidence -> B's decoder (separately in each
half, as in the real match) -> the official scorer, via rondo.harness.score.row. Two ball
conditions are scored. "full" is the provider's complete ball track. "corrupt" removes the ball
wherever the REAL match's evidence has it missing (its tracking-availability pattern, which is
input data, never a label), because that is the condition the submission actually runs under.

Candidate features are ball-free team-shape signals (FEATURES below). Each enters as a log-odds
shift on the H/A split of B's own emission, so B's decoder is neither duplicated nor edited:

    logit(share') = logit(share) + sum_k weight_k * feature_k

The weights are fitted ONCE, by logistic regression of our own match's control evidence on the
features (self-supervised, allowed by RULES.md: no label of this match is read), and frozen
before any external match is scored. The keep-or-drop call is rondo.harness.score.keep -- the
harness's rule, not ours.
"""

import argparse
import warnings

import numpy as np

from rondo import decoder
from rondo.harness import datasets, labels, score
from rondo.signals import external
from rondo.signals.m2_evidence import AWAY, HOME, build_evidence, team_spread_ratio

EPS = 1e-6
SMOOTH = 25        # frames, centred window, same as shape_spread
MIN_PLAYERS = 8    # per team, as shape_spread
LINE_PLAYERS = 4   # the back line: the 4 deepest outfield players


# -- features ----------------------------------------------------------------------------------

def smooth(raw, window=SMOOTH):
    """NaN-aware centred moving average; NaN stays NaN."""
    ok = np.isfinite(raw)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, raw, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    n, half = len(raw), window // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[hi] - csum[lo]) / (ccnt[hi] - ccnt[lo])
    out[~ok] = np.nan
    return out


def defends_low(player_xy, player_team, half, length):
    """{(half, team): True if that team defends the x = 0 goal}, from its mean x in the half."""
    out = {}
    for number in sorted(set(half.tolist()) - {0}):
        rows = half == number
        for team in (HOME, AWAY):
            out[(number, team)] = bool(np.nanmean(player_xy[rows][:, player_team == team, 0])
                                       < length / 2.0)
    return out


def line_height(player_xy, player_team, half, pitch):
    """(line_H - line_A) / 10 m: how much higher H's back line stands than A's.

    A team's line is the mean distance from its own goal of its LINE_PLAYERS deepest outfield
    players (the deepest player, normally the keeper, is skipped). The team in possession pushes
    its line up and the defending team drops, so positive should lean H. NaN where a team has
    fewer than MIN_PLAYERS tracked.
    """
    length = pitch[0]
    low = defends_low(player_xy, player_team, half, length)
    lines = []
    for team in (HOME, AWAY):
        x = player_xy[:, player_team == team, 0]
        own = np.full(x.shape, np.nan)
        for number in sorted(set(half.tolist()) - {0}):
            rows = half == number
            own[rows] = x[rows] if low[(number, team)] else length - x[rows]
        depth = np.sort(own, axis=1)  # NaN sorts last
        line = depth[:, 1:1 + LINE_PLAYERS].mean(axis=1)
        line[np.isfinite(own).sum(axis=1) < MIN_PLAYERS] = np.nan
        lines.append(line)
    return smooth((lines[HOME] - lines[AWAY]) / 10.0)


def marking_gap(player_xy, player_team, chunk=20000):
    """log(free_H / free_A): free_T is T's mean distance to the nearest opponent.

    The attacking team runs into space and the defending team marks, so the team in possession
    is the freer one: positive should lean H. NaN where a team has fewer than MIN_PLAYERS tracked.
    """
    h, a = player_xy[:, player_team == HOME], player_xy[:, player_team == AWAY]
    n = len(player_xy)
    free = np.full((n, 2), np.nan)
    for start in range(0, n, chunk):
        hh, aa = h[start:start + chunk], a[start:start + chunk]
        dist = np.sqrt(((hh[:, :, None, :] - aa[:, None, :, :]) ** 2).sum(-1))  # [n, H, A]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN, as wanted
            near_h = np.nanmin(dist, axis=2)  # each H player's nearest A
            near_a = np.nanmin(dist, axis=1)
            free[start:start + chunk, 0] = np.nanmean(near_h, axis=1)
            free[start:start + chunk, 1] = np.nanmean(near_a, axis=1)
    count_h = np.isfinite(h[..., 0]).sum(1)
    count_a = np.isfinite(a[..., 0]).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.log(free[:, 0] / free[:, 1])
    raw[(count_h < MIN_PLAYERS) | (count_a < MIN_PLAYERS) | ~np.isfinite(raw)] = np.nan
    return smooth(raw)


def centred(feature, half):
    """Subtract each half's median, so a team's style (always wider, always higher) is not read
    as a possession prior. Uses no label: only the feature itself."""
    out = feature.copy()
    for number in sorted(set(half.tolist()) - {0}):
        rows = half == number
        out[rows] -= np.nanmedian(feature[rows])
    return out


def features(player_xy, player_team, half, pitch):
    """{name: float[N]} every candidate, NaN where unknown."""
    spread = team_spread_ratio(player_xy, player_team)
    line = line_height(player_xy, player_team, half, pitch)
    mark = marking_gap(player_xy, player_team)
    return {"shape": spread, "shape_c": centred(spread, half), "line_c": centred(line, half),
            "mark_c": centred(mark, half), "pspeed": player_speed(player_xy)}


def player_speed(player_xy, fps=25):
    """Mean speed of every tracked player, m/s, smoothed over 1 s. Low in stoppages."""
    step = np.linalg.norm(np.diff(player_xy, axis=0), axis=2) * fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = np.concatenate(([np.nan], np.nanmean(step, axis=1)))
    return smooth(raw)


def slow_dead(ev, pspeed, slow, level, scope="missing"):
    """dead_ball raised to `level` where it is unknown (0.5), players are slower than `slow` m/s
    and -- for scope "missing" -- the ball is not seen: on our match a missing ball with slow
    players is a stoppage far more often than not."""
    out = dict(ev)
    p = np.asarray(ev["dead_ball"], dtype=float).copy()
    hit = (p == 0.5) & (np.nan_to_num(pspeed, nan=np.inf) < slow)
    if scope == "missing":
        hit &= ev["ball_missing"]
    p[hit] = level
    out["dead_ball"] = p
    return out


def lead(feature, half, frames):
    """The feature read `frames` AHEAD within each half (decoding is offline, so the future is
    allowed); the last `frames` of each half have nothing ahead and become NaN."""
    out = np.full(len(feature), np.nan)
    for number in sorted(set(half.tolist()) - {0}):
        rows = np.flatnonzero(half == number)
        start, stop = int(rows.min()), int(rows.max()) + 1
        if stop - start > frames:
            out[start:stop - frames] = feature[start + frames:stop]
    return out


LEADS_S = tuple(range(0, 13))  # seconds tried when fitting a lead


def auc(values, positive):
    """Area under the ROC curve of `values` for `positive` (NaN ignored)."""
    from scipy.stats import rankdata
    ok = np.isfinite(values)
    v, p = values[ok], positive[ok]
    n1, n0 = int(p.sum()), int((~p).sum())
    if not n1 or not n0:
        return float("nan")
    ranks = rankdata(v)
    return float((ranks[p].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit_lead(feature, half, ev, keep, fps=25):
    """Seconds ahead at which the feature best separates our own control evidence (no label)."""
    use, y = control_target(ev, keep)
    scores = {s: auc(lead(feature, half, s * fps)[use], y[use] > 0.5) for s in LEADS_S}
    return max(scores, key=lambda s: (abs(scores[s] - 0.5), -s)), scores


# The experiments. Each is {"features": names whose fitted log-odds are added, "gate": dead_ball
# level at or above which control is switched off, "params": decoder parameter overrides}.
# "shape" is the raw shape_spread as M2 ships it (the Nov 13 test); the "_c" features are centred
# per half; "_lead" reads the centred feature ahead by the lead fitted on our own match.
VARIANTS = {
    "base": {},
    "shape": {"features": ("shape",)},
    "shape_c": {"features": ("shape_c",)},
    "line_c": {"features": ("line_c",)},
    "mark_c": {"features": ("mark_c",)},
    "combo_c": {"features": ("shape_c", "line_c", "mark_c")},
    "shape_lead": {"features": ("shape_c_lead",)},
    "mark_lead": {"features": ("mark_c_lead",)},
    "combo_lead": {"features": ("shape_c_lead", "mark_c_lead")},
    "gate70": {"gate": 0.7},
    "gate80": {"gate": 0.8},
    "gate100": {"gate": 1.0},
    # The decoder change proposed to B (Nov 13): no-evidence frames stop leaning dead, plus two
    # transition/emission settings, chosen on Metrica across corruptions, confirmed on IDSSE.
    "decoder_fix": {"dead_prior": True, "params": {"p_to_dead": 0.004, "touch_weight": 0.2}},
}
LEAD_OF = ("shape_c", "mark_c")


def gate_control(ev, level):
    """Control switched off wherever dead_ball >= level: a player holding the ball for a
    throw-in or standing over a free kick is not possession in play."""
    out = dict(ev)
    dead = ev["dead_ball"] >= level
    out["control_H"] = ev["control_H"] & ~dead
    out["control_A"] = ev["control_A"] & ~dead
    return out


# -- fitting and decoding ----------------------------------------------------------------------

def fit_weights(X, y):
    """Logistic regression slopes (intercept fitted, then dropped) by Newton's method."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        grad = X.T @ (y - p)
        hess = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(beta)), grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta[1:]


def control_target(ev, keep):
    """(mask, y): frames where exactly one team has sustained control, y = 1 for H."""
    use = keep & (ev["control_H"] != ev["control_A"])
    return use, ev["control_H"].astype(float)


def fit_variant(names, feats, ev, keep):
    """{feature: weight} for one variant, fitted on control evidence where all are known."""
    if not names:
        return {}
    use, y = control_target(ev, keep)
    X = np.column_stack([feats[name] for name in names])
    use = use & np.isfinite(X).all(axis=1)
    return dict(zip(names, fit_weights(X[use], y[use]).tolist()))


def log_odds(feats, weights):
    total = np.zeros(len(next(iter(feats.values()))))
    for name, weight in weights.items():
        total += np.where(np.isfinite(feats[name]), weight * feats[name], 0.0)
    return total


def shift_share(loglik, delta):
    """Add `delta` to the log-odds of H vs A, leaving the dead-ball probability alone."""
    out = loglik.copy()
    h, a = np.exp(loglik[:, 0]), np.exp(loglik[:, 1])
    share = np.clip(h / (h + a), EPS, 1 - EPS)
    new = 1.0 / (1.0 + np.exp(-(np.log(share / (1 - share)) + delta)))
    out[:, 0] += np.log(np.clip(new, EPS, 1 - EPS) / share)
    out[:, 1] += np.log(np.clip(1 - new, EPS, 1 - EPS) / (1 - share))
    return out


def decode_match(ev, half, delta=None, params=decoder.DEFAULTS):
    """Codes for every row. Each half decoded alone from D; rows outside a half stay X."""
    codes = np.full(len(half), labels.X, dtype=np.int8)
    loglik = decoder.emission_loglik(ev, params)
    if delta is not None:
        loglik = shift_share(loglik, delta)
    log_trans = decoder.log_transitions(params)
    log_init = np.log([EPS, EPS, 1.0])
    for number in sorted(set(half.tolist()) - {0}):
        rows = np.flatnonzero(half == number)
        start, stop = int(rows.min()), int(rows.max()) + 1
        codes[start:stop] = decoder.viterbi(loglik[start:stop], log_trans, log_init)
    return codes


def corruption_mask(real_missing, n_frames):
    """Ball-missing mask for an external match: the real match's pattern in its graded halves."""
    pattern = np.concatenate([real_missing[a:b] for a, b in decoder.graded_blocks()])
    return np.resize(pattern, n_frames)


def occlusion_mask(ball_xy, player_xy, real_missing, contact_keep, seed=0, radius=1.5, pad=6):
    """Ball-missing mask that hides the ball around player contacts, as video occlusion does.

    Each run of frames where a player is within `radius` of the ball is hidden (padded by `pad`
    frames each side) with probability `contact_keep`; then frames of the real match's missing
    pattern are added until the total missing share equals the real one. Seeded, so the same
    arguments always give the same mask.
    """
    from rondo.signals.m1_ingest import _runs
    rng = np.random.default_rng(seed)
    n = len(ball_xy)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        near = np.nanmin(np.sqrt(((player_xy - ball_xy[:, None, :]) ** 2).sum(-1)), axis=1)
    mask = np.zeros(n, dtype=bool)
    for start, stop in _runs(near < radius):
        if rng.random() < contact_keep:
            mask[max(0, start - pad):min(n, stop + pad)] = True
    target = corruption_mask(real_missing, n)
    need = target.mean() - mask.mean()
    if need > 0:
        extra = target & ~mask
        frac = need / max(extra.mean(), 1e-9)
        # Add whole runs of the real pattern, in order, until the share is reached.
        for start, stop in _runs(extra):
            if rng.random() < frac:
                mask[start:stop] = True
    return mask


def real_half(n_frames):
    """Half number per frame of OUR match (1, 2 in the graded halves, 0 elsewhere)."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, (start, stop) in enumerate(decoder.graded_blocks(), start=1):
        half[start:stop] = number
    return half


def prepare_real():
    """Our own match, M1 + M2 exactly as the submission builds them, plus its features and the
    feature leads fitted on its control evidence. Reads no label."""
    from games import GAME, GAMES
    from load_smart_tagging import load as load_smart_tagging
    from rondo.signals.m1_ingest import main as ingest_atd
    from rondo.signals.m2_evidence import KICKOFF_FRAMES
    m1 = ingest_atd(GAME)
    ev, _ = build_evidence(m1["ball_xy"], m1["player_xy"], m1["player_team"],
                           GAMES[GAME]["pitch"], clean=False,
                           smart_tagging=load_smart_tagging(GAME), kickoffs=KICKOFF_FRAMES)
    half = real_half(len(ev["ball_missing"]))
    feats = features(m1["player_xy"], m1["player_team"], half, GAMES[GAME]["pitch"])
    leads = {}
    for name in LEAD_OF:
        leads[name], _ = fit_lead(feats[name], half, ev, half > 0)
        feats[name + "_lead"] = lead(feats[name], half, leads[name] * 25)
    return {"ev": ev, "half": half, "feats": feats, "leads": leads, "m1": m1}


def subsample(ball_xy, every):
    """The ball seen only every `every` frames and linearly interpolated in between, as a
    low-rate detector would give it; frames that were missing stay missing."""
    t = np.arange(len(ball_xy))
    seen = np.isfinite(ball_xy[:, 0])
    keep = seen & (t % every == 0)
    out = np.full_like(ball_xy, np.nan)
    for axis in range(2):
        out[:, axis] = np.interp(t, t[keep], ball_xy[keep, axis])
    out[~seen] = np.nan
    return out


# Ways to make an external ball look like ours. We could not pin down which one our ATD matches
# (each fixes one statistic and breaks another), so a decoder change has to hold across all.
#   corrupt   the real match's missing pattern, transplanted at random times
#   occl50    half of all player contacts hidden (video occlusion), topped up to the real share
#   low8      the ball sampled every 8 frames (about 3 Hz), then the real missing pattern
def _corrupt(truth, meta, player_xy, real):
    ball = truth["ball_xy"].copy()
    ball[corruption_mask(real["ev"]["ball_missing"], len(ball))] = np.nan
    return ball


def _occl50(truth, meta, player_xy, real):
    ball = truth["ball_xy"].copy()
    ball[occlusion_mask(ball, player_xy, real["ev"]["ball_missing"], 0.5)] = np.nan
    return ball


def _low8(truth, meta, player_xy, real):
    ball = subsample(truth["ball_xy"], 8)
    ball[corruption_mask(real["ev"]["ball_missing"], len(ball))] = np.nan
    return ball


def _deadmiss(truth, meta, player_xy, real, dead_rate=0.78, alive_rate=0.13, seed=0):
    """The ball hidden the way Metrica's own tracking loses it: whole stoppages go missing
    (78% of dead frames in their native sample) and a few contacts in play (13% of alive frames).
    Our match is from the same provider and its missing ball sits in long stretches with slow
    players, i.e. in stoppages, which the random "corrupt" pattern does not reproduce. Uses the
    EXTERNAL match's labels to place the gaps; never ours."""
    from rondo.signals.m1_ingest import _runs
    rng = np.random.default_rng(seed)
    ball = truth["ball_xy"].copy()
    codes = truth["codes"]
    dead = codes == labels.D
    mask = np.zeros(len(ball), dtype=bool)
    for start, stop in _runs(dead):
        if rng.random() < dead_rate:
            mask[start:stop] = True
    alive = (codes == labels.H) | (codes == labels.A)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        near = np.nanmin(np.sqrt(((player_xy - ball[:, None, :]) ** 2).sum(-1)), axis=1)
    contacts = [(a, b) for a, b in _runs(alive & (near < 1.5))]
    order = rng.permutation(len(contacts))
    goal = alive_rate * alive.sum()
    hidden = 0
    for i in order:
        if hidden >= goal:
            break
        a, b = contacts[i]
        a, b = max(0, a - 12), min(len(ball), b + 12)
        hidden += int((alive[a:b] & ~mask[a:b]).sum())
        mask[a:b] |= alive[a:b]
    ball[mask] = np.nan
    return ball


CORRUPTIONS = {"corrupt": _corrupt, "occl50": _occl50, "low8": _low8, "deadmiss": _deadmiss,
               "full": lambda truth, meta, player_xy, real: truth["ball_xy"].copy()}


def dead_prior(ev):
    """dead_ball remapped so 0.5 ("no idea") gives D, H and A equal emission weight.

    B's emission scores D at p and H, A at (1 - p) * share. With p = 0.5 and share = 0.5 that is
    0.5 against 0.25 each, so every frame without evidence leans dead. p' = p / (p + 2 (1 - p))
    sends 0.5 to 1/3, keeps 0 and 1 fixed and keeps the order of every other level.
    """
    out = dict(ev)
    p = np.asarray(ev["dead_ball"], dtype=float)
    out["dead_ball"] = p / (p + 2.0 * (1.0 - p))
    return out


def prepare_matches(real, selection=None, conditions=("corrupt", "full")):
    """[(dataset, match, gt, half, feats, {condition: evidence})] for the selected matches.

    `selection` holds dataset names and/or (dataset, match) pairs; None means every match.
    `conditions` are names in CORRUPTIONS.
    """
    out = []
    for dataset, match_id in score.available():
        if selection and (dataset, match_id) not in selection and dataset not in selection:
            continue
        gt, truth, meta = score.load(dataset, match_id)
        truth = dict(truth, codes=gt)  # only the corruption reads it, to place the gaps
        player_xy, player_team = external.players(dataset, match_id)
        feats = features(player_xy, player_team, truth["half"], tuple(meta["pitch"]))
        for name, seconds in real["leads"].items():
            feats[name + "_lead"] = lead(feats[name], truth["half"], seconds * 25)
        evs = {}
        for condition in conditions:
            ball = CORRUPTIONS[condition](truth, meta, player_xy, real)
            evs[condition], _ = build_evidence(ball, player_xy, player_team, tuple(meta["pitch"]))
        out.append((dataset, match_id, gt, truth["half"], feats, evs))
    return out


def run(specs, prepared, real, scale=1.0, bootstrap=0, log=print):
    """{experiment: [rows]} for every spec on every prepared match and condition."""
    weights = {name: {k: v * scale for k, v in
                      fit_variant(spec.get("features", ()), real["feats"], real["ev"],
                                  real["half"] > 0).items()}
               for name, spec in specs.items()}
    results = {}
    for dataset, match_id, gt, half, feats, evs in prepared:
        for condition, ev in evs.items():
            for name, spec in specs.items():
                used = gate_control(ev, spec["gate"]) if spec.get("gate") is not None else ev
                if spec.get("slow_dead"):
                    used = slow_dead(used, feats["pspeed"], **spec["slow_dead"])
                if spec.get("dead_prior"):
                    used = dead_prior(used)
                w = weights[name]
                params = dict(decoder.DEFAULTS, **spec.get("params", {}))
                pred = decode_match(used, half, log_odds(feats, w) if w else None, params)
                experiment = "rondo_{}_{}".format(name, condition)
                notes = " ".join("{}={:.4f}".format(k, v) for k, v in w.items())
                row = score.row(experiment, dataset, match_id, gt, pred, bootstrap, notes)
                results.setdefault(experiment, []).append(row)
                if log:
                    log("{:<24} {:<8} {:<14} spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}".format(
                        experiment, dataset, match_id, row["spell_f1_mean"], row["macro_f1"],
                        row["n_pred_spells"], row["n_gt_spells"]))
    return results


def summary(results, baseline="base", log=print):
    """Keep-or-drop of every experiment against the baseline, per condition and dataset."""
    names = sorted({e[len("rondo_"):].rsplit("_", 1)[0] for e in results})
    conditions = sorted({e.rsplit("_", 1)[1] for e in results})
    for condition in conditions:
        for scope, pick in (("held-out IDSSE", "idsse"), ("Metrica anchor", "metrica")):
            base = [r for r in results.get("rondo_{}_{}".format(baseline, condition), [])
                    if r["dataset"] == pick]
            if not base:
                continue
            for name in names:
                if name == baseline:
                    continue
                rows = [r for r in results["rondo_{}_{}".format(name, condition)]
                        if r["dataset"] == pick]
                kept, delta = score.keep(base, rows)
                log("{:<8} {:<15} {:<14} base {:.4f}  variant {:.4f}  delta {:+.4f}  "
                    "spells {}/{} -> {}".format(
                        condition, scope, name, score.mean_spell_f1(base),
                        score.mean_spell_f1(rows), delta,
                        sum(r["n_pred_spells"] for r in rows),
                        sum(r["n_gt_spells"] for r in rows), "KEEP" if kept else "drop"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", choices=datasets.DATASETS)
    parser.add_argument("--match", action="append")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS))
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every fitted weight (choose it on Metrica, not IDSSE)")
    parser.add_argument("--condition", action="append", choices=sorted(CORRUPTIONS),
                        help="ball conditions to score (default: corrupt, deadmiss, full)")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--sheet", default=None)
    args = parser.parse_args(argv)
    specs = {name: VARIANTS[name] for name in (args.variant or VARIANTS)}
    specs.setdefault("base", {})

    real = prepare_real()
    selection = list(args.dataset or [])
    selection += [pair for pair in score.available() if args.match and pair[1] in args.match]
    prepared = prepare_matches(real, selection or None,
                               tuple(args.condition or ("corrupt", "deadmiss", "full")))
    results = run(specs, prepared, real, args.scale, args.bootstrap)
    score.write_rows([r for rows in results.values() for r in rows], args.sheet)
    print()
    summary(results)


if __name__ == "__main__":
    main()
