"""M2 -> M3 emission contract: what Signals (A) hands the Decoder (B).

One array per key, indexed by TACTICAL VIDEO FRAME (row i = submission row i, length 199319).
The tactical ATD is on the video timeline: video_frame = frame_id - 1 (starter/load_atd.py).
games.id_to_video is for the ground truth's separate IdFrame counter, not the ATD. Frames with no
tracking (before kickoff, halftime, after the whistle) carry the missing value.

Handoff file: np.savez_compressed("evidence.npz", **evidence).
"""

import numpy as np

# key -> (kind, missing value). ball_xy is (n, 2) metres on the 105x68 pitch; the rest are (n,).
# restart_team / smart_tag_team are float so NaN can mean "unknown": 0 = H, 1 = A.
COLUMNS = {
    "ball_xy": ("float", np.nan),
    "ball_missing": ("bool", True),
    "touch_H": ("bool", False),
    "touch_A": ("bool", False),
    "control_H": ("bool", False),
    "control_A": ("bool", False),
    "duel_zone": ("bool", False),
    "dead_ball": ("prob", 0.5),
    "restart_team": ("team", np.nan),
    "smart_tag_dead": ("prob", 0.5),
    "smart_tag_team": ("team", np.nan),
}


def empty(n_frames):
    """Evidence that says nothing: every key at its missing value."""
    out = {}
    for key, (kind, missing) in COLUMNS.items():
        shape = (n_frames, 2) if key == "ball_xy" else (n_frames,)
        dtype = bool if kind == "bool" else float
        out[key] = np.full(shape, missing, dtype=dtype)
    return out


def check(evidence, n_frames):
    """Raise ValueError naming every way `evidence` breaks the contract. Returns it unchanged."""
    problems = []
    missing = sorted(set(COLUMNS) - set(evidence))
    if missing:
        problems.append("missing keys {}".format(missing))
    for key, (kind, _) in COLUMNS.items():
        if key not in evidence:
            continue
        value = np.asarray(evidence[key])
        shape = (n_frames, 2) if key == "ball_xy" else (n_frames,)
        if value.shape != shape:
            problems.append("{} has shape {}, expected {}".format(key, value.shape, shape))
            continue
        if kind == "bool" and value.dtype != bool:
            problems.append("{} must be bool, got {}".format(key, value.dtype))
        if kind == "prob" and (np.isnan(value).any() or (value < 0).any() or (value > 1).any()):
            problems.append("{} must be in [0, 1] with no NaN (missing = 0.5)".format(key))
        if kind == "team":
            known = value[~np.isnan(value)]
            if not np.isin(known, (0, 1)).all():
                problems.append("{} must be 0, 1 or NaN".format(key))
    if problems:
        raise ValueError("evidence breaks the M2->M3 contract: " + "; ".join(problems))
    return evidence
