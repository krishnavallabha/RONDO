"""Evidence -> decoder -> validated submission CSV.

    python -m rondo.run --dummy --out out/dummy.csv
    python -m rondo.run --evidence out/evidence.npz --out out/RONDO.csv
"""

import argparse

import numpy as np

from games import GAME, GAMES
from rondo import contract, decoder, dummy, submission


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence", help="npz saved by M2 (see rondo/contract.py)")
    source.add_argument("--dummy", action="store_true", help="synthetic evidence, no data needed")
    parser.add_argument("--out", required=True, help="CSV to write")
    args = parser.parse_args(argv)

    n_frames = GAMES[GAME]["n_video_frames"]
    if args.dummy:
        ev, _ = dummy.evidence(n_frames)
    else:
        with np.load(args.evidence) as data:
            ev = contract.check({key: data[key] for key in data.files}, n_frames)

    labels = decoder.decode(ev, n_frames)
    print("wrote", submission.write(labels, args.out))


if __name__ == "__main__":
    main()
