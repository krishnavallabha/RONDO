"""M0 -- hidden-channel inspection.

Checks whether the tactical EPTS export carries extra channels (ball_state, ball_owning_team)
that might hold Metrica's own possession signal. Prints only aggregate statistics, never raw
rows, so the output is safe to paste into notes.

This lives OUTSIDE the organisers' repo on purpose. It imports their starter loader from there.

Run (PowerShell):

    $env:MTAH_DATA_DIR = "C:\\Hackathon\\barca\\data"
    cd C:\\Hackathon\\barca\\RONDO
    python -m rondo.signals.m0_inspect

Nothing here writes data or labels.
"""

import os
import sys

import rondo.signals  # noqa: F401  (puts the challenge repo and its starter/ on sys.path)
from load_atd import load  # noqa: E402  (organisers' starter loader, same alignment rules)

CANDIDATES = ("ball_state", "ball_owning_team")


def main():
    dataset = load()
    df = dataset.to_df()
    print("columns: {}".format(list(df.columns)))
    print("rows: {}".format(len(df)))

    for name in CANDIDATES:
        if name not in df.columns:
            print("\n{}: NOT PRESENT in this export".format(name))
            continue
        col = df[name]
        filled = col.notna().mean()
        print("\n{}: filled {:.1%} of rows".format(name, filled))
        counts = col.value_counts(dropna=True).head(10)
        print("  top values (count):")
        for value, count in counts.items():
            print("    {!r}: {}".format(value, int(count)))
        # A possession signal should change rarely (hand-overs), not every frame.
        valid = col.dropna()
        flips = int((valid != valid.shift()).sum() - 1) if len(valid) > 1 else 0
        print("  value changes between consecutive filled frames: {}".format(flips))


if __name__ == "__main__":
    main()
