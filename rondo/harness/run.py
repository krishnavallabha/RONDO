"""The whole pipeline on an external match, with no gated data anywhere: the harness run.

    external truth.npz (ball) + players  ->  corruption model  ->  M2 build_evidence
                                         ->  M3 decode_half    ->  official scorer -> sheet

This is the Nov 13 harness run, and the one thing it refuses to need is this match's data. The
other external path, rondo/signals/m2_validate.py, transplants the real ball's missing mask and
fits feature weights on the real match, so it only runs where the gated bundle is unpacked. This
one takes its noise from a committed profile (rondo/harness/noise.py), so a clean clone can
reproduce every number, CI could run it, and a teammate without the data is not blocked.

    python -m rondo.harness run                               # clean + corrupted, every match
    python -m rondo.harness run --profile metrica_native --bootstrap 400
    python -m rondo.harness run --dataset metrica --seed 1

Each row lands in the results sheet under `harness_<condition>`, so the Nov 14 sweeps compare
against it with score.keep like any other experiment.
"""

import numpy as np

from rondo import contract, decoder
from rondo.harness import datasets, labels, noise, score
from rondo.signals import external
from rondo.signals.export import to_contract
from rondo.signals.m2_evidence import build_evidence

CLEAN = "clean"  # the provider's own ball, uncorrupted: the ceiling this match cannot reach


def evidence_for(dataset, match_id, profile=None, seed=0, external_dir=None):
    """(evidence, half, gt_codes, meta) for one external match under one ball condition.

    `profile` None means the provider's ball as it arrived. Otherwise the ball is corrupted
    first, so M2 sees the kind of track this match actually has.
    """
    gt, truth, meta = score.load(dataset, match_id, external_dir)
    player_xy, player_team = external.players(dataset, match_id, external_dir)
    ball = truth["ball_xy"]
    if profile is not None:
        ball = noise.apply(ball, profile, seed)
    # clean=True: M1's teleport rejection and gap bridging run here exactly as they do on the
    # real match, so what the decoder sees has been through the same cleaning.
    ev, _ = build_evidence(ball, player_xy, player_team, tuple(meta["pitch"]))
    # Contract-checked for the same reason the real pipeline checks it: the decoder reads these
    # keys assuming NaN means unknown, and M2's internal -1 leaking through once already cost a
    # debugging session (see rondo/decoder.emission_loglik).
    ev = contract.check(to_contract(ev), len(gt))
    return ev, truth["half"], gt, meta


def decode(ev, half, params=decoder.DEFAULTS):
    """H/A/D per frame, each half decoded on its own from its own start, X outside the halves."""
    codes = np.full(len(half), labels.X, dtype=np.int8)
    loglik = decoder.emission_loglik(ev, params)
    for number in sorted(set(np.asarray(half).tolist()) - {0}):
        rows = np.flatnonzero(half == number)
        start, stop = int(rows.min()), int(rows.max()) + 1
        codes[start:stop] = decoder.decode_half(loglik[start:stop], params)
    return codes


def one(dataset, match_id, profile=None, condition=CLEAN, seed=0, params=decoder.DEFAULTS,
        bootstrap_n=0, external_dir=None):
    """Score the pipeline on one external match under one condition. Returns a sheet row."""
    ev, half, gt, _ = evidence_for(dataset, match_id, profile, seed, external_dir)
    pred = decode(ev, half, params)
    notes = condition if profile is None else "{} seed={} missing={:.3f}".format(
        condition, seed, float(np.mean(ev["ball_missing"])))
    return score.row("harness_" + condition, dataset, match_id, gt, pred, bootstrap_n, notes)


def run(matches=None, profiles=None, seed=0, params=decoder.DEFAULTS, bootstrap_n=0,
        external_dir=None, sheet=None, log=print):
    """Every condition on every match. `profiles` is {condition: profile or None}."""
    matches = matches or score.available(external_dir)
    profiles = profiles or {CLEAN: None}
    rows = []
    for dataset, match_id in matches:
        for condition, profile in profiles.items():
            row = one(dataset, match_id, profile, condition, seed, params, bootstrap_n,
                      external_dir)
            rows.append(row)
            if log:
                log("{:<22} {:<8} {:<14} spellF1 {:.4f}  macroF1 {:.4f}  spells {}/{}".format(
                    row["experiment"], dataset, match_id, row["spell_f1_mean"], row["macro_f1"],
                    row["n_pred_spells"], row["n_gt_spells"]))
    if rows:
        score.write_rows(rows, sheet, external_dir)
    return rows


# The organisers' published baseline on the full match: raw per-frame nearest-player possession,
# no model. Our Nov 13 milestone is to beat it on held-out external data, not on this match.
BASELINE_SPELL_F1 = 0.2433


def report(rows, target=BASELINE_SPELL_F1, log=print):
    """Mean spellF1 per condition and dataset, against the baseline we have to beat."""
    conditions = sorted({row["experiment"] for row in rows})
    for experiment in conditions:
        for scope, pick in (("held-out IDSSE", "idsse"), ("Metrica anchor", "metrica")):
            subset = [row for row in rows
                      if row["experiment"] == experiment and row["dataset"] == pick]
            if not subset:
                continue
            mean = score.mean_spell_f1(subset)
            worst = min(row["spell_f1_mean"] for row in subset)
            log("{:<22} {:<15} {} matches  spellF1 mean {:.4f}  worst {:.4f}  {} {:.4f}".format(
                experiment, scope, len(subset), mean, worst,
                "ABOVE" if mean > target else "below", target))
    return conditions


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="python -m rondo.harness run",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", action="append", choices=datasets.DATASETS)
    parser.add_argument("--match", action="append")
    parser.add_argument("--profile", action="append",
                        help="noise profile name or path (default: metrica_native); "
                             "repeat to score several conditions")
    parser.add_argument("--clean", action="store_true",
                        help="also score the provider's uncorrupted ball")
    parser.add_argument("--seed", type=int, default=0, help="corruption seed")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--sheet")
    parser.add_argument("--external-dir", default=None, help="override data/external")
    args = parser.parse_args(argv)

    matches = [pair for pair in score.available(args.external_dir)
               if (not args.dataset or pair[0] in args.dataset)
               and (not args.match or pair[1] in args.match)]
    if not matches:
        print("no external matches built; run: python -m rondo.harness fetch && "
              "python -m rondo.harness labels")
        return 1

    profiles = {}
    for name in args.profile or ["metrica_native"]:
        profiles[name.split("/")[-1].replace(".json", "")] = noise.load(name)
    if args.clean or not profiles:
        profiles[CLEAN] = None

    rows = run(matches, profiles, args.seed, bootstrap_n=args.bootstrap,
               external_dir=args.external_dir, sheet=args.sheet)
    print()
    report(rows)
    return 0
