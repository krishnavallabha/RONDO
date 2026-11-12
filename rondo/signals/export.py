"""Run M0-M2 and write the evidence file the decoder reads.

    python -m rondo.signals.export --out out/evidence.npz
"""

import argparse
import os

import numpy as np

from games import GAME, GAMES
from rondo import contract
from rondo.signals import m2_evidence

TEAM_KEYS = ("restart_team", "smart_tag_team")


def to_contract(ev):
    """M2's int8 teams (-1 unknown) become the contract's float teams (NaN unknown)."""
    out = dict(ev)
    for key in TEAM_KEYS:
        team = np.asarray(ev[key])
        out[key] = np.where(team < 0, np.nan, team).astype(float)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("out", "evidence.npz"))
    args = parser.parse_args(argv)

    n_frames = GAMES[GAME]["n_video_frames"]
    evidence = contract.check(to_contract(m2_evidence.main()), n_frames)

    folder = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(args.out, **evidence)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
