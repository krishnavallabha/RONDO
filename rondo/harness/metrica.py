"""Metrica sample matches -> per-frame H/A/D.

Metrica's sample data has no per-frame possession channel, only hand-coded events, so the labels
here are DERIVED. That is the point of this converter: Metrica wrote both these events and this
challenge's ground truth, so the rules below are our best reading of their conventions, and any
place where our decoder disagrees with these labels is a place where it would disagree with the
real ground truth too. IDSSE (rondo/harness/idsse.py) is the opposite case -- provider labels,
nothing derived -- which is why the keep-or-drop decision rule is applied there and conventions
are learned here.

The rules, in full (they are the thing to argue with, not the code):

* Possession is LAST TOUCH. A pass in flight still belongs to the team that played it, so a
  turnover lands on the opponent's RECOVERY / CHALLENGE-WON frame, not on the earlier BALL LOST
  frame of the team that lost it. Metrica's own event pairs are written that way: BALL LOST at
  1110-1134 by Away, RECOVERY at 1134 by Home.
* The ball goes dead at the frame the stoppage physically happens -- the END frame of a BALL OUT
  (when it crosses the line, not when it was kicked), the END frame of a goal-scoring shot, the
  frame of a FAULT RECEIVED -- and comes alive again on the START frame of the next SET PIECE,
  which is the restart kick itself. A restart's kick frame is alive and belongs to the restarting
  team.
* A period that ends mid-stoppage stays dead to its last tracked frame.
* Frames with no tracking row are X: nothing to score there.

Frames are 1-based in Metrica's files and 0-based everywhere in this module, which is the
indexing a submission uses.
"""

import os

import numpy as np
import pandas as pd

from rondo.harness import datasets, labels

PITCH = (105.0, 68.0)  # metres; Metrica normalises both axes to [0, 1] of these

# Events that say who has the ball, applied from their start frame. BALL OUT is in here because
# its team is the last team to touch the ball, so it keeps possession while the ball runs out.
OWNER_TYPES = ("PASS", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE", "BALL OUT")

# Events that stop play, and which of their frames the stoppage starts on.
STOP_AT_END = ("BALL OUT",)          # the frame the ball leaves the field
STOP_AT_START = ("FAULT RECEIVED",)  # the frame of the foul

COLUMNS = {"Team": "team", "Type": "kind", "Subtype": "subtype", "Period": "period",
           "Start Frame": "start", "End Frame": "end"}


def read_events(path):
    """The event CSV, normalised: safe column names, upper-case text, 0-based frames."""
    frame = pd.read_csv(path)[list(COLUMNS)].rename(columns=COLUMNS)
    for column in ("team", "kind", "subtype"):
        frame[column] = frame[column].fillna("").astype(str).str.upper().str.strip()
    for column in ("start", "end"):
        frame[column] = frame[column].astype(np.int64) - 1
    return frame


def read_tracking(path):
    """(period, frame index, ball xy in [0, 1]) from one Metrica tracking CSV.

    Three header rows: team name, shirt numbers, then the real column names. The ball is the
    last pair of columns, and is NaN wherever their tracking lost it -- which is the one number
    the corruption model wants out of this file.
    """
    frame = pd.read_csv(path, skiprows=2)
    return (frame["Period"].to_numpy(dtype=np.int64),
            frame["Frame"].to_numpy(dtype=np.int64) - 1,
            frame.iloc[:, -2:].to_numpy(dtype=float))


def is_goal(kind, subtype):
    """A goal, told apart from a 'GOAL KICK' restart: the token must be exactly GOAL."""
    return kind in ("SHOT", "BALL OUT") and "GOAL" in subtype.split("-")


def challenge_owner(code, subtype, other):
    """Who holds the ball after a CHALLENGE, or None when the subtype does not say."""
    tokens = subtype.split("-")
    if "WON" in tokens:
        return code
    if "LOST" in tokens:
        return other
    return None


def owner_changes(events):
    """[(frame, code)] possession changes from one period's events, in frame order."""
    out = []
    for row in events.itertuples(index=False):
        if row.team == "HOME":
            code, other = labels.H, labels.A
        elif row.team == "AWAY":
            code, other = labels.A, labels.H
        else:
            continue
        if row.kind in OWNER_TYPES:
            out.append((int(row.start), code))
        elif row.kind == "CHALLENGE":
            won = challenge_owner(code, row.subtype, other)
            if won is not None:
                out.append((int(row.start), won))
    return sorted(out)


def dead_spans(events):
    """[(start, stop or None)] half-open dead-ball spans from one period's events.

    A stoppage opens at the physical stop and closes on the next restart kick; stop is None when
    no restart follows, meaning the period ended during it. Overlapping stoppages (a foul given
    while the ball is already out) collapse into one span, because two stops with no restart
    between them are one dead ball.
    """
    stops, restarts = [], []
    for row in events.itertuples(index=False):
        if row.kind in STOP_AT_END or is_goal(row.kind, row.subtype):
            stops.append(max(int(row.start), int(row.end)))
        elif row.kind in STOP_AT_START:
            stops.append(int(row.start))
        if row.kind == "SET PIECE":
            restarts.append(int(row.start))

    restarts = np.array(sorted(restarts), dtype=np.int64)
    spans = []
    for stop in sorted(stops):
        index = int(np.searchsorted(restarts, stop, side="left"))
        end = int(restarts[index]) if index < len(restarts) else None
        if spans and (spans[-1][1] is None or stop <= spans[-1][1]):
            spans[-1] = (spans[-1][0], None if end is None else max(spans[-1][1] or 0, end))
        else:
            spans.append((stop, end))
    return spans


def build(match_id, external_dir=None):
    """(codes, truth, meta) for one Metrica sample match. Needs the files on disk."""
    external_dir = external_dir or datasets.EXTERNAL_DIR
    events_path, home_path = datasets.require(
        "metrica", match_id, ("events", "tracking_home"), external_dir)
    events = read_events(events_path)
    period, index, ball = read_tracking(home_path)

    n_frames = int(index.max()) + 1
    codes = np.full(n_frames, labels.X, dtype=np.int8)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=float)
    half = np.zeros(n_frames, dtype=np.int8)
    ball_xy[index] = ball * PITCH
    half[index] = period

    for number in sorted(set(period.tolist())):
        rows = index[period == number]
        start, stop = int(rows.min()), int(rows.max()) + 1
        block = events[events["period"] == number]

        changes = owner_changes(block)
        if not changes:
            raise ValueError("no possession events in period {} of {}".format(number, match_id))
        # Possession first -- each change holds until the next one -- then stoppages on top,
        # because a dead ball overrides whoever was last in control.
        edges = [frame for frame, _ in changes[1:]] + [stop]
        codes[start:min(changes[0][0], stop)] = changes[0][1]  # the team that kicks off
        for (left, code), right in zip(changes, edges):
            codes[max(start, left):min(right, stop)] = code
        for dead_start, dead_end in dead_spans(block):
            codes[max(start, dead_start):stop if dead_end is None else min(stop, dead_end)] = \
                labels.D

    truth = {"ball_xy": ball_xy, "half": half,
             "ball_missing": np.isnan(ball_xy[:, 0]),
             "frame_n": np.arange(1, n_frames + 1, dtype=np.int64)}
    meta = {
        "dataset": "metrica",
        "match_id": match_id,
        "fps": datasets.FPS,
        "pitch": list(PITCH),
        "label_source": "derived from Metrica event CSV (rules in rondo/harness/metrica.py)",
        "n_events": int(len(events)),
        "native_ball_missing": round(float(np.isnan(ball_xy[:, 0]).mean()), 4),
        "periods": sorted(int(p) for p in set(period.tolist())),
    }
    meta.update(labels.summary(codes, datasets.FPS))
    return codes, truth, meta


def write(match_id, external_dir=None):
    """Build one match and save labels.csv + truth.npz + meta.json. Returns (path, meta)."""
    external_dir = external_dir or datasets.EXTERNAL_DIR
    codes, truth, meta = build(match_id, external_dir)
    folder = labels.match_dir(external_dir, "metrica", match_id)
    return labels.write(folder, codes, truth, meta), meta


def built(match_id, external_dir=None):
    folder = labels.match_dir(external_dir or datasets.EXTERNAL_DIR, "metrica", match_id)
    return os.path.isfile(os.path.join(folder, "labels.csv"))
