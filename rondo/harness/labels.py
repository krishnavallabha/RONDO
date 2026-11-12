"""The H/A/D/X label file, shared by every external match.

External matches are stored in exactly the shape a submission has -- a frame,label CSV indexed
from 0, read back with the official possession_metrics.read_label_csv -- so the harness scores
them with the same code path that scores the real thing, and a bug in one shows up in the other.

Alongside each labels.csv sits truth.npz (ball position, half, provider frame numbers) and
meta.json (what the match is, how the labels were made, what they look like). The npz is what
the corruption model reads on Nov 13; the json is what the results sheet quotes.
"""

import json
import os

import numpy as np
import pandas as pd

import possession_metrics as pm

LABELS = np.array(["H", "A", "D", "X"])  # index = code, same order as pm.LABELS
H, A, D, X = 0, 1, 2, 3


def match_dir(external_dir, dataset, match_id):
    return os.path.join(external_dir, "labels", dataset, match_id)


def write(folder, codes, truth=None, meta=None):
    """Write labels.csv (+ optional truth.npz, meta.json). Returns the labels.csv path.

    The CSV is written exactly like a submission: '\\n' line endings, frame 0..n-1, and then
    read back through the official validator, so an external match that the scorer could not
    load fails here rather than halfway through an experiment.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "labels.csv")
    pd.DataFrame({"frame": np.arange(len(codes), dtype=np.int64),
                  "label": LABELS[np.asarray(codes, dtype=np.int8)]}).to_csv(
        path, index=False, lineterminator="\n")

    reason = pm.validate_submission(pm.read_label_csv(path), len(codes))
    if reason:
        raise ValueError("{} is not a valid label file: {}".format(path, reason))
    if truth:
        np.savez_compressed(os.path.join(folder, "truth.npz"), **truth)
    if meta:
        with open(os.path.join(folder, "meta.json"), "w", newline="\n") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return path


def read(folder):
    """Label codes of a built match, as int8 in pm's alphabet."""
    path = os.path.join(folder, "labels.csv")
    if not os.path.isfile(path):
        raise IOError("no labels at {}\n\nRun: python -m rondo.harness labels".format(path))
    return pm.label_codes(pm.read_label_csv(path))


def read_meta(folder):
    with open(os.path.join(folder, "meta.json")) as handle:
        return json.load(handle)


def summary(codes, fps=25):
    """What a built match looks like -- the numbers that catch a broken converter.

    A plausible football match is roughly 55-65% alive, splits the alive frames somewhere near
    45/55 between the teams, and has a few hundred possessions averaging well under a minute.
    Anything far outside that is a label bug, not a strange match.
    """
    codes = np.asarray(codes)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    graded = int((codes != X).sum())
    alive = int(((codes == H) | (codes == A)).sum())
    lengths = [stop - start for start, stop, _ in spells]
    return {
        "n_frames": int(len(codes)),
        "n_graded": graded,
        "alive_share": round(alive / graded, 4) if graded else 0.0,
        "home_share_alive": round(float((codes == H).sum()) / alive, 4) if alive else 0.0,
        "n_possessions": len(spells),
        "mean_possession_s": round(float(np.mean(lengths)) / fps, 2) if lengths else 0.0,
        "median_possession_s": round(float(np.median(lengths)) / fps, 2) if lengths else 0.0,
        "n_dead_segments": sum(1 for _, _, code in pm.segments(codes, runs) if code == D),
    }
