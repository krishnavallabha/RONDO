"""CSV writer that refuses to leave a file the official scorer would reject."""

import os

import numpy as np
import pandas as pd

import possession_metrics as pm
from rondo.decoder import graded_blocks


def write(labels, path):
    """Write frame,label with '\\n' line endings (byte-identical on every OS), then re-check it."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels), dtype=np.int64), "label": labels}).to_csv(
        path, index=False, lineterminator="\n")

    reason = pm.validate_submission(pm.read_label_csv(path), len(labels))
    if reason:
        raise ValueError("{} fails official validation: {}".format(path, reason))
    for start, stop in graded_blocks():
        if (np.asarray(labels[start:stop]) == "X").any():
            raise ValueError("X inside graded frames {}..{}".format(start, stop - 1))
    return path
