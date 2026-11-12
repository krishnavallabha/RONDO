"""Run M1-M2 and write the evidence file the decoder reads.

    python -m rondo.signals.export --out out/evidence.npz

Any other match (the harness): build plain arrays, then

    from rondo.signals.m2_evidence import build_evidence
    from rondo.signals.export import save
    evidence, info = build_evidence(ball_xy, player_xy, player_team, pitch=(105.0, 68.0))
    save(evidence, "out/external/<match>.npz")
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


def save(evidence, path):
    """Contract-check evidence from build_evidence() (any match) and write it as .npz."""
    out = contract.check(to_contract(evidence), len(evidence["ball_missing"]))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("out", "evidence.npz"))
    args = parser.parse_args(argv)

    evidence = m2_evidence.main()
    if len(evidence["ball_missing"]) != GAMES[GAME]["n_video_frames"]:
        raise ValueError("evidence length does not match the tactical video")
    print("wrote", save(evidence, args.out))


if __name__ == "__main__":
    main()
