"""The official scorer, pointed at external matches, writing one row per (experiment, match).

Everything here goes through possession_metrics.evaluate -- the organisers' file, imported, never
reimplemented. A metric RONDO reports is a metric they compute, so a number from the harness and
a number from the leaderboard mean the same thing.

    python -m rondo.harness score --pred out/dummy.csv --match Sample_Game_1
    python -m rondo.harness score --baseline constant_home --bootstrap 400

The results sheet (data/external/results.csv by default) is append-only and keyed by experiment
name, so a sweep is a sequence of score() calls and the sheet is the record of what was tried.
"""

import os

import numpy as np
import pandas as pd

import possession_metrics as pm
from rondo.harness import datasets, idsse, labels, metrica

# The noise band the keep-or-drop rule uses. It is the scorer's own constant, not ours.
NOISE_BAND = pm.SPELL_NOISE_BAND

# Tolerance (frames) whose latency we report as boundary error. 25 = one second at 25 fps, the
# looser of the scorer's two tolerances, so matched boundaries are matched for the right reason.
BOUNDARY_TAU = 25

SHEET = "results.csv"
COLUMNS = ["experiment", "dataset", "match", "n_frames", "n_graded",
           "spell_f1_mean", "spell_f1_0.3", "spell_f1_0.4", "spell_f1_0.5", "spell_f1_0.6",
           "spell_f1_0.7", "spell_boot_low", "spell_boot_high", "bootstrap_n",
           "macro_f1", "f1_H", "f1_A", "f1_D", "alive_dead_f1", "accuracy",
           "n_gt_spells", "n_pred_spells", "spell_count_error",
           "boundary_median_frames", "boundary_p90_frames", "transition_f1",
           "gt_home_share", "pred_home_share", "notes"]


def match_folder(dataset, match_id, external_dir=None):
    return labels.match_dir(external_dir or datasets.EXTERNAL_DIR, dataset, match_id)


def ensure_labels(dataset, match_id, external_dir=None, log=print):
    """Build a match's labels if they are not there yet. Returns its folder."""
    folder = match_folder(dataset, match_id, external_dir)
    if not os.path.isfile(os.path.join(folder, "labels.csv")):
        log("building labels for {} {}".format(dataset, match_id))
        module = idsse if dataset == "idsse" else metrica
        module.write(match_id, external_dir)
    return folder


def load(dataset, match_id, external_dir=None):
    """(codes, truth dict, meta dict) for a built external match."""
    folder = ensure_labels(dataset, match_id, external_dir)
    with np.load(os.path.join(folder, "truth.npz")) as data:
        truth = {key: data[key] for key in data.files}
    return labels.read(folder), truth, labels.read_meta(folder)


def available(external_dir=None):
    """[(dataset, match)] whose source files are downloaded -- what the harness can score now."""
    return [(dataset, match_id) for dataset, match_id in datasets.every_match()
            if datasets.have(dataset, match_id,
                             ("events", "tracking_home") if dataset == "metrica" else None,
                             external_dir)]


def row(experiment, dataset, match_id, gt_codes, pred_codes, bootstrap_n=0, notes=""):
    """One results-sheet row: the official metrics for one prediction on one match."""
    if len(pred_codes) != len(gt_codes):
        raise ValueError("{}: prediction has {} frames, {} has {}".format(
            experiment, len(pred_codes), match_id, len(gt_codes)))
    report = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    boundary = report["transitions"][BOUNDARY_TAU]
    out = {
        "experiment": experiment,
        "dataset": dataset,
        "match": match_id,
        "n_frames": report["n_frames"],
        "n_graded": report["n_graded"],
        "spell_f1_mean": report["spell_f1_mean"],
        "spell_boot_low": report["spell_bootstrap_low"],
        "spell_boot_high": report["spell_bootstrap_high"],
        "bootstrap_n": bootstrap_n,
        "macro_f1": report["macro_f1"],
        "alive_dead_f1": report["alive_dead_f1"],
        "accuracy": report["accuracy"],
        "n_gt_spells": report["n_gt_spells"],
        "n_pred_spells": report["n_pred_spells"],
        "spell_count_error": report["spell_count_error"],
        # Median |latency| of a matched boundary is the number the snapping work on Nov 14 has to
        # move; mean latency hides a two-sided error, so it is not the one reported here.
        "boundary_median_frames": (abs(boundary["median_latency"])
                                   if boundary["median_latency"] is not None else None),
        "boundary_p90_frames": boundary["p90_abs_latency"],
        "transition_f1": boundary["f1"],
        "gt_home_share": report["gt_home_share"],
        "pred_home_share": report["pred_home_share"],
        "notes": notes,
    }
    for threshold, scores in report["spell_f1"].items():
        out["spell_f1_{}".format(threshold)] = scores["f1"]
    for name in ("H", "A", "D"):
        out["f1_{}".format(name)] = report["per_class"][name]["f1"]
    return out


def write_rows(rows, path=None, external_dir=None):
    """Append rows to the results sheet, creating it with the full header if it is new."""
    external_dir = external_dir or datasets.EXTERNAL_DIR
    path = path or os.path.join(external_dir, SHEET)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    header = not os.path.isfile(path)
    frame.to_csv(path, mode="a" if not header else "w", header=header, index=False,
                 lineterminator="\n")
    return path


def score(experiment, predict, matches=None, bootstrap_n=0, external_dir=None, sheet=None,
          notes="", log=print):
    """Run one experiment over external matches and append its rows to the sheet.

    `predict` is called as predict(truth, meta, gt_codes) and returns one code per frame. It gets
    the ground-truth codes because the reference baselines here are defined as corruptions of the
    truth -- a real experiment reads truth["ball_xy"] and the corruption model instead, and must
    not look at gt_codes.
    """
    rows = []
    for dataset, match_id in matches or available(external_dir):
        gt_codes, truth, meta = load(dataset, match_id, external_dir)
        pred = np.asarray(predict(truth, meta, gt_codes), dtype=np.int8)
        rows.append(row(experiment, dataset, match_id, gt_codes, pred, bootstrap_n, notes))
        if log:
            log("{:<22} {:<8} {:<14} spellF1 {:.4f}  macroF1 {:.4f}".format(
                experiment, dataset, match_id, rows[-1]["spell_f1_mean"], rows[-1]["macro_f1"]))
    if rows:
        write_rows(rows, sheet, external_dir)
    return rows


def score_csv(experiment, pred_path, dataset, match_id, bootstrap_n=0, external_dir=None,
              sheet=None, notes="", log=print):
    """Score a submission-shaped CSV against one external match."""
    codes = pm.label_codes(pm.read_label_csv(pred_path))
    return score(experiment, lambda truth, meta, gt: codes, [(dataset, match_id)],
                 bootstrap_n, external_dir, sheet, notes or os.path.basename(pred_path), log)


def mean_spell_f1(rows):
    """Mean spellF1 over the matches of one experiment -- what the decision rule compares."""
    values = [r["spell_f1_mean"] for r in rows]
    return float(np.mean(values)) if values else 0.0


def keep(before, after, band=NOISE_BAND):
    """(keep?, delta) for the decision rule: a change survives only if it beats the noise band.

    Nothing about this is a close call on purpose. A gain inside the band is a gain we cannot
    tell from resampling noise, so it is dropped however sensible the idea was.
    """
    delta = mean_spell_f1(after) - mean_spell_f1(before)
    return bool(delta > band), delta


# -- reference predictors ----------------------------------------------------------------------
#
# Not models. They exist so the harness has something to score before the decoder can run on
# external data, and so the sheet records what a floor and a near-ceiling look like on these
# matches -- without which a first real number is impossible to read.

def constant_home(truth, meta, gt_codes):
    """Everything H. The degenerate floor spellF1 is supposed to punish."""
    return np.full(len(gt_codes), labels.H, dtype=np.int8)


def truth_shifted(frames):
    """The truth with every boundary moved `frames` late: what a timing error alone costs.

    This is the calibration behind the Nov 14 boundary-snapping decision -- it says how many
    frames of systematic lag spellF1 tolerates before the possession stops matching at each IoU.
    """
    def predict(truth, meta, gt_codes):
        return np.concatenate([np.full(frames, gt_codes[0], dtype=np.int8),
                               gt_codes[:-frames]]) if frames else gt_codes.copy()
    return predict


BASELINES = {
    "constant_home": constant_home,
    "truth": truth_shifted(0),
    "truth_lag_12": truth_shifted(12),
    "truth_lag_25": truth_shifted(25),
    "truth_lag_50": truth_shifted(50),
}
