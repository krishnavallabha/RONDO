"""Run M1-M2 and write the evidence file the decoder reads.

    python -m rondo.signals.export --out out/evidence.npz
    python -m rondo.signals.export --out out/evidence.npz --no-m7     # heuristic evidence only

By default the file also carries p_dead and p_home from the learned emissions
(rondo/signals/m8_stoppage.py; --model m7 for rondo/signals/m7_learned.py, the cf22e581
submission); the decoder uses them when they are present.

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
from rondo.signals import m1_ingest, m2_evidence

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
    parser.add_argument("--no-m7", action="store_true", help="leave out the learned p_dead / p_home")
    parser.add_argument("--model", choices=["m8", "m7"], default="m8",
                        help="learned emissions: m8 (stoppage experts, default) or m7 (cf22e581)")
    args = parser.parse_args(argv)

    m1 = m1_ingest.main(GAME)
    evidence = m2_evidence.main(m1_data=m1)
    if len(evidence["ball_missing"]) != GAMES[GAME]["n_video_frames"]:
        raise ValueError("evidence length does not match the tactical video")
    if not args.no_m7:
        from rondo.signals.m2_validate import real_half
        half = real_half(len(evidence["ball_missing"]))
        if args.model == "m8":
            from rondo.signals import m8_stoppage
            evidence["p_dead"], evidence["p_home"] = m8_stoppage.probabilities(
                evidence, m1["player_xy"], m1["player_team"], half, tuple(GAMES[GAME]["pitch"]))
        else:
            from rondo.signals import m7_learned
            evidence["p_dead"], evidence["p_home"] = m7_learned.probabilities(
                evidence, m1["player_xy"], m1["player_team"], half)
    print("wrote", save(evidence, args.out))


if __name__ == "__main__":
    main()
