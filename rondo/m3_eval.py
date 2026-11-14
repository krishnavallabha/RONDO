"""Decoder (M3) experiments on the external matches, with evidence built once and cached.

    python -m rondo.m3_eval                                  # every experiment in EXPERIMENTS
    python -m rondo.m3_eval --exp base --exp min75

M2 evidence is the slow part and does not depend on the decoder, so it is built once per match
and ball condition under data/external/cache/m3/, then every decoder variant reuses it.

Protocol (docs/EXTERNAL_DATA.md): choose on the 2 Metrica matches (same provider as ours),
confirm on the 7 IDSSE matches, keep a change only if it beats the 0.04 noise band there.

Ball conditions -- the external ball made to look like ours:
  deadmiss  whole stoppages hidden plus some contacts, as the provider's tracking loses the ball
  noisy     deadmiss, plus 1.6 m of slow per-player position noise: our ball sits ~1.5 m from
            the players even at a kick (A, Nov 13), which clean external tracking never shows
"""

import argparse
import os

import numpy as np

import possession_metrics as pm
from rondo import decoder
from rondo.harness import datasets, score
from rondo.signals import external
from rondo.signals.export import to_contract
from rondo.signals.m2_evidence import build_evidence
from rondo.signals.m2_validate import _deadmiss, smooth

CACHE = os.path.join(datasets.EXTERNAL_DIR, "cache", "m3")
CONDITIONS = ("deadmiss", "noisy")
SCATTER_M = 1.6
SCATTER_SMOOTH = 25  # frames: the offset drifts over about a second, it does not flicker


def scatter(player_xy, seed=0):
    """player_xy plus slow Gaussian noise of SCATTER_M metres per axis, per player."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=player_xy.shape)
    for p in range(player_xy.shape[1]):
        for axis in range(2):
            noise[:, p, axis] = smooth(noise[:, p, axis], SCATTER_SMOOTH)
    noise *= SCATTER_M / noise.reshape(-1, 2).std(axis=0)
    return player_xy + noise


def evidence(dataset, match_id, condition):
    """(evidence, half, gt_codes), cached on disk after the first build."""
    path = os.path.join(CACHE, "{}_{}_{}.npz".format(dataset, match_id, condition))
    if os.path.isfile(path):
        with np.load(path) as data:
            items = {key: data[key] for key in data.files}
        return ({k[3:]: v for k, v in items.items() if k.startswith("ev_")},
                items["half"], items["gt"])
    gt, truth, meta = score.load(dataset, match_id)
    player_xy, player_team = external.players(dataset, match_id)
    ball = _deadmiss(dict(truth, codes=gt), meta, player_xy, None)
    if condition == "noisy":
        player_xy = scatter(player_xy)
    ev = to_contract(build_evidence(ball, player_xy, player_team, tuple(meta["pitch"]))[0])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, half=truth["half"], gt=gt,
                        **{"ev_" + key: value for key, value in ev.items()})
    return ev, truth["half"], gt


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


EXPERIMENTS = {
    "base": {},
    "min25": {"min_alive": 25},
    "min50": {"min_alive": 50},
    "min75": {"min_alive": 75},
    "min100": {"min_alive": 100},
    "min75_dead50": {"min_alive": 75, "min_dead": 50},
    "min75_mbr": {"min_alive": 75, "mbr": True},
}


def run(names, log=print):
    """{(name, dataset, condition): [spellF1 per match]}, logged as it goes."""
    out = {}
    for dataset, match_id in score.available():
        for condition in CONDITIONS:
            ev, half, gt = evidence(dataset, match_id, condition)
            runs = pm.valid_runs(gt)
            for name in names:
                params = dict(decoder.DEFAULTS, **EXPERIMENTS[name])
                result = pm.spell_scores(gt, decode_codes(ev, half, params), runs)
                out.setdefault((name, dataset, condition), []).append(result["mean_f1"])
                log("{:<14} {:<8} {:<14} {:<9} spellF1 {:.4f}  spells {}/{}".format(
                    name, dataset, match_id, condition, result["mean_f1"],
                    result["n_pred"], result["n_gt"]))
    return out


def summary(out, names, log=print):
    log("\nmean spellF1 (delta vs base; keep only if IDSSE delta > 0.04)")
    for dataset in datasets.DATASETS:
        for condition in CONDITIONS:
            base = np.mean(out[("base", dataset, condition)])
            cells = ["{} {:.4f} ({:+.4f})".format(
                name, np.mean(out[(name, dataset, condition)]),
                np.mean(out[(name, dataset, condition)]) - base) for name in names]
            log("{:<8} {:<9} ".format(dataset, condition) + "  ".join(cells))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", action="append", choices=sorted(EXPERIMENTS))
    args = parser.parse_args(argv)
    names = ["base"] + [n for n in (args.exp or EXPERIMENTS) if n != "base"]
    summary(run(names), names)


if __name__ == "__main__":
    main()
