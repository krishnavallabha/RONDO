"""Harness command line: get the external matches, label them, score against them.

    python -m rondo.harness status                      what is downloaded and built
    python -m rondo.harness fetch                       ~2.6 GB, mostly IDSSE positional XML
    python -m rondo.harness labels                      external data -> labels.csv per match
    python -m rondo.harness score --baseline all --bootstrap 400
    python -m rondo.harness score --pred out/dummy.csv --match Sample_Game_1
"""

import argparse
import os
import sys

from rondo.harness import datasets, idsse, labels, metrica, score


def add_selection(parser):
    parser.add_argument("--dataset", choices=datasets.DATASETS, action="append",
                        help="limit to one dataset (repeatable)")
    parser.add_argument("--match", action="append", help="limit to one match id (repeatable)")
    parser.add_argument("--external-dir", default=None, help="override data/external")


def selected(args):
    """[(dataset, match)] the flags ask for, whether or not it is downloaded yet."""
    wanted = args.dataset or list(datasets.DATASETS)
    return [(dataset, match_id) for dataset, match_id in datasets.every_match()
            if dataset in wanted and (not args.match or match_id in args.match)]


def cmd_status(args):
    external_dir = args.external_dir or datasets.EXTERNAL_DIR
    print(external_dir)
    for dataset, match_id in selected(args):
        source = "downloaded" if datasets.have(dataset, match_id) else "missing"
        module = idsse if dataset == "idsse" else metrica
        folder = labels.match_dir(external_dir, dataset, match_id)
        if module.built(match_id, external_dir):
            meta = labels.read_meta(folder)
            built = "{} frames, {} possessions, {:.0%} alive".format(
                meta["n_frames"], meta["n_possessions"], meta["alive_share"])
        else:
            built = "no labels"
        print("  {:<8} {:<14} {:<11} {}".format(dataset, match_id, source, built))
    return 0


def cmd_fetch(args):
    datasets.fetch_all(args.dataset or datasets.DATASETS, matches=args.match,
                       force=args.force, external_dir=args.external_dir)
    return 0


def cmd_labels(args):
    external_dir = args.external_dir or datasets.EXTERNAL_DIR
    missing = 0
    for dataset, match_id in selected(args):
        module = idsse if dataset == "idsse" else metrica
        if not datasets.have(dataset, match_id,
                             ("events", "tracking_home") if dataset == "metrica" else None,
                             external_dir):
            print("  {:<8} {:<14} not downloaded, skipped".format(dataset, match_id))
            missing += 1
            continue
        if module.built(match_id, external_dir) and not args.force:
            print("  {:<8} {:<14} already built".format(dataset, match_id))
            continue
        path, meta = module.write(match_id, external_dir)
        print("  {:<8} {:<14} {} frames, {:.0%} alive, {} possessions, home {:.0%} -> {}".format(
            dataset, match_id, meta["n_frames"], meta["alive_share"], meta["n_possessions"],
            meta["home_share_alive"], os.path.relpath(path, external_dir)))
    return 1 if missing else 0


def cmd_score(args):
    external_dir = args.external_dir or datasets.EXTERNAL_DIR
    matches = [pair for pair in selected(args) if pair in score.available(external_dir)]
    if not matches:
        print("no external matches available; run: python -m rondo.harness fetch")
        return 1

    if args.pred:
        if len(matches) != 1:
            print("--pred scores one match: pass --match")
            return 1
        dataset, match_id = matches[0]
        score.score_csv(args.experiment or "pred", args.pred, dataset, match_id,
                        args.bootstrap, external_dir, args.sheet)
        return 0

    names = list(score.BASELINES) if args.baseline in (None, "all") else [args.baseline]
    for name in names:
        score.score(args.experiment or name, score.BASELINES[name], matches,
                    args.bootstrap, external_dir, args.sheet)
    print("sheet:", args.sheet or os.path.join(external_dir, score.SHEET))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m rondo.harness",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="what is downloaded and built")
    add_selection(status)
    status.set_defaults(run=cmd_status)

    fetch = sub.add_parser("fetch", help="download external matches")
    add_selection(fetch)
    fetch.add_argument("--force", action="store_true", help="re-download even if present")
    fetch.set_defaults(run=cmd_fetch)

    build = sub.add_parser("labels", help="convert external data to H/A/D labels")
    add_selection(build)
    build.add_argument("--force", action="store_true", help="rebuild even if present")
    build.set_defaults(run=cmd_labels)

    scoring = sub.add_parser("score", help="score a prediction against external matches")
    add_selection(scoring)
    scoring.add_argument("--pred", help="a submission-shaped CSV to score (needs --match)")
    scoring.add_argument("--baseline", choices=sorted(score.BASELINES) + ["all"],
                         help="score a reference predictor instead (default: all)")
    scoring.add_argument("--experiment", help="name for the results sheet row")
    scoring.add_argument("--bootstrap", type=int, default=0,
                         help="bootstrap draws for the spellF1 interval (400 is plenty)")
    scoring.add_argument("--sheet", help="results CSV to append to")
    scoring.set_defaults(run=cmd_score)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
