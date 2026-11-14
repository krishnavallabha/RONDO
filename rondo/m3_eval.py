"""Decoder (M3) experiments on the external matches, with M7 trained out-of-fold.

    python -m rondo.m3_eval                                  # every experiment in EXPERIMENTS
    python -m rondo.m3_eval --exp m7 --exp m7_mbr

Protocol (docs/EXTERNAL_DATA.md): choose on the 2 Metrica matches (same provider as ours),
confirm on the 7 IDSSE matches, keep a change only if it beats the 0.04 noise band there.

Conditions are M7's five (rondo.signals.m7_learned.CONDITIONS): full, deadmiss, deadmiss plus
1.0 / 1.6 m of slow player noise, and our missing-ball mask plus 1.3 m. Player noise drifts over
4 s (m7_learned.slow_noise), so player speeds stay realistic. The Nov 14 morning version of this
file smoothed over 1 s, which made players run at ~14 m/s and inflated the cost of noise.

M7 is refit per fold so a match is never scored by a model that saw it: Metrica held out as a
block, IDSSE in three folds (sorted ids, i % 3), the same folds as A's out-of-fold numbers.
Features are built once per match and condition and cached under data/external/cache/m3v2/.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np

import possession_metrics as pm
from rondo import decoder
from rondo.harness import datasets, score
from rondo.signals import external
from rondo.signals import m2_evidence as M
from rondo.signals import m2_validate as V
from rondo.signals import m7_learned as M7
from rondo.signals.export import to_contract

CACHE = os.path.join(datasets.EXTERNAL_DIR, "cache", "m3v2")
CONDITIONS = tuple(M7.CONDITIONS)
REALISTIC = ("n16dm", "n13corrupt")  # the two closest to our ball; a change must not lose here


def _path(dataset, match_id, condition, kind):
    return os.path.join(CACHE, "{}_{}_{}_{}.npz".format(dataset, match_id, condition, kind))


def build(args):
    """Evidence and M7 features for one match under every condition, cached (worker)."""
    dataset, match_id, real_missing = args
    if all(os.path.isfile(_path(dataset, match_id, c, "x")) for c in CONDITIONS):
        return
    gt, truth, meta = score.load(dataset, match_id)
    truth = dict(truth, codes=gt)
    player_xy, player_team = external.players(dataset, match_id)
    real = {"ev": {"ball_missing": real_missing}}
    for cond, (corruption, sigma) in M7.CONDITIONS.items():
        ball = V.CORRUPTIONS[corruption](truth, meta, player_xy, real)
        players = player_xy if sigma == 0 else player_xy + np.stack(   # same seeds as M7 training
            [M7.slow_noise(len(ball), sigma, 10 + i) for i in range(player_xy.shape[1])], 1)
        ev, _ = M.build_evidence(ball, players, player_team, tuple(meta["pitch"]))
        XT, XD, _, _ = M7.features(ev, players, player_team, truth["half"])
        os.makedirs(CACHE, exist_ok=True)
        np.savez(_path(dataset, match_id, cond, "x"), XT=XT.astype(np.float32),
                 XD=XD.astype(np.float32), half=truth["half"], gt=gt,
                 **{"ev_" + k: v for k, v in to_contract(ev).items()})


def load(dataset, match_id, condition):
    with np.load(_path(dataset, match_id, condition, "x")) as data:
        items = {key: data[key] for key in data.files}
    ev = {k[3:]: v for k, v in items.items() if k.startswith("ev_")}
    return ev, items


def folds():
    """[[(dataset, match_id)]]: Metrica as one block, IDSSE in three."""
    matches = sorted(score.available())
    out = [[m for m in matches if m[0] == "metrica"]]
    idsse = [m for m in matches if m[0] != "metrica"]
    out += [idsse[i::3] for i in range(3)]
    return [f for f in out if f]


def oof_probabilities(log=print):
    """Cache (p_dead, p_home) for every match and condition from a model that never saw it."""
    for held in folds():
        if all(os.path.isfile(_path(d, m, c, "p")) for d, m in held for c in CONDITIONS):
            continue
        rows = []
        for d, m in sorted(set(score.available()) - set(held)):
            for c in CONDITIONS:
                _, it = load(d, m, c)
                rows.append(M7.training_rows(it["XT"], it["XD"], it["gt"], it["half"]))
        team, dead = M7.fit(rows)
        del rows
        log("fold fitted, held out: {}".format(", ".join(m for _, m in held)))
        for d, m in held:
            for c in CONDITIONS:
                _, it = load(d, m, c)
                np.savez(_path(d, m, c, "p"),
                         p_dead=1 / (1 + np.exp(-M7.logit(dead, it["XD"].astype(np.float64)))),
                         p_home=1 / (1 + np.exp(-M7.logit(team, it["XT"].astype(np.float64)))))


def decode_codes(ev, half, params):
    """Codes for every row, each half decoded alone; X outside the halves."""
    codes = np.full(len(half), pm.X, dtype=np.int8)
    loglik = decoder.emission_loglik(ev, params)
    for number in sorted(set(half.tolist()) - {0}):
        rows = np.flatnonzero(half == number)
        start, stop = int(rows.min()), int(rows.max()) + 1
        half_ll = loglik[start:stop]
        codes[start:stop] = (decoder.mbr_half(half_ll, params) if params.get("mbr")
                             else decoder.decode_half(half_ll, params))
    return codes


# "m7": True = use the out-of-fold M7 probabilities; otherwise the heuristic emissions.
EXPERIMENTS = {
    "heur": {"m7": False},
    "m7": {},
    "m7_mbr": {"mbr": True},
    "m7_min50": {"min_alive": 50},
    "m7_min100": {"min_alive": 100},
    "m7_min150": {"min_alive": 150},
    "m7_dead50": {"min_dead": 50},
    "m7_direct5e5": {"p_switch_direct": 5e-5},
    "m7_direct1e3": {"p_switch_direct": 1e-3},
    "m7_prior30": {"m7_prior_dead": 0.30},
    "m7_prior55": {"m7_prior_dead": 0.55},
}
BASE = "m7"


def score_match(args):
    """[(name, dataset, condition, spellF1, n_pred, n_gt)] for one match (worker)."""
    dataset, match_id, names = args
    out = []
    for cond in CONDITIONS:
        ev, it = load(dataset, match_id, cond)
        with np.load(_path(dataset, match_id, cond, "p")) as p:
            m7 = dict(ev, p_dead=p["p_dead"], p_home=p["p_home"])
        gt, runs = it["gt"], pm.valid_runs(it["gt"])
        for name in names:
            spec = dict(EXPERIMENTS[name])
            use_m7 = spec.pop("m7", True)
            params = dict(decoder.DEFAULTS, **spec)
            r = pm.spell_scores(gt, decode_codes(m7 if use_m7 else ev, it["half"], params), runs)
            out.append((name, dataset, match_id, cond, r["mean_f1"], r["n_pred"], r["n_gt"]))
    return out


def summary(results, names, log=print):
    log("\nmean spellF1 per condition, delta vs {} (keep only if > 0.04 and no loss on {})".format(
        BASE, "/".join(REALISTIC)))
    for name in names:
        for dataset in datasets.DATASETS:
            cells, deltas, split = [], [], []
            for cond in CONDITIONS:
                mine = [r for r in results if r[0] == name and r[1] == dataset and r[3] == cond]
                base = [r for r in results if r[0] == BASE and r[1] == dataset and r[3] == cond]
                if not mine:
                    continue
                f1 = np.mean([r[4] for r in mine])
                deltas.append(f1 - np.mean([r[4] for r in base]))
                split.append(sum(r[5] for r in mine) / sum(r[6] for r in mine))
                cells.append("{} {:.3f}".format(cond, f1))
            if cells:
                log("{:<13} {:<8} mean {:.3f} ({:+.3f})  pred/true spells {:.2f}   {}".format(
                    name, dataset, np.mean([float(c.split()[1]) for c in cells]), np.mean(deltas),
                    np.mean(split), "  ".join(cells)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", action="append", choices=sorted(EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    names = [BASE] + [n for n in (args.exp or EXPERIMENTS) if n != BASE]
    real_missing = np.asarray(M.main()["ball_missing"]).copy()   # label-free, as M7 training
    matches = sorted(score.available())
    with Pool(args.workers) as pool:
        pool.map(build, [(d, m, real_missing) for d, m in matches])
        print("features cached")
        oof_probabilities()
        results = [r for part in pool.imap_unordered(score_match, [(d, m, names) for d, m in matches])
                   for r in part]
    for r in sorted(results):
        print("{:<13} {:<8} {:<14} {:<11} spellF1 {:.4f}  spells {}/{}".format(*r))
    summary(results, names)


if __name__ == "__main__":
    main()
