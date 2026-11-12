"""IDSSE / Sportec matches -> per-frame H/A/D, straight from the provider's own channels.

The positional XML carries two attributes on every ball frame:

    BallStatus       1 = alive, 0 = dead
    BallPossession   1 = home team, 2 = guest team

so these labels are read, not inferred, which is what makes these seven matches the held-out set
the keep-or-drop rule is applied to. (Metrica's sample is the opposite: derived labels, used to
learn conventions -- see rondo/harness/metrica.py.)

The 1 = home mapping is the DFL convention, and it is checked rather than trusted: build() scores
the per-frame labels against the independent event file and refuses to write a match where the
two disagree more often than they agree, which is exactly what a flipped mapping looks like.

Parsing is a line scan, not an XML tree: the files are 350-420 MB with one element per line, and
an ElementTree walk over ~3 million player frames costs minutes to reach the ~270k ball frames
that matter. A ball frame is one carrying BallStatus, so the scan does not depend on how the
provider names the ball's FrameSet.
"""

import datetime as dt
import os
import re
import xml.etree.ElementTree as ElementTree

import numpy as np

from rondo.harness import datasets, labels

FRAMESET = re.compile(r"<FrameSet\b([^>]*)>")
ATTRS = re.compile(r'([\w:-]+)="([^"]*)"')
HALVES = ("firstHalf", "secondHalf")

# A flipped home/away mapping shows up as agreement far below a half. Real agreement runs high
# (the events and the tracking describe the same match), so anything in between is a parser bug
# and should stop the build rather than quietly feed the harness.
MIN_EVENT_AGREEMENT = 0.6


def read_matchinfo(path):
    """Team ids and names, pitch size, and what the match is called."""
    root = ElementTree.parse(path).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    pitch = (float(environment.get("PitchX", 105.0)), float(environment.get("PitchY", 68.0)))
    return {
        "match_id": general.get("MatchId"),
        "title": general.get("MatchTitle"),
        "season": general.get("Season"),
        "home_team": general.get("HomeTeamName"),
        "away_team": general.get("GuestTeamName"),
        "home_team_id": general.get("HomeTeamId"),
        "away_team_id": general.get("GuestTeamId"),
        "result": general.get("Result"),
        "pitch": list(pitch),
    }


def epoch(text):
    """ISO timestamp -> seconds since the epoch, UTC. The files mix +00:00 and +02:00 offsets."""
    return dt.datetime.fromisoformat(text).timestamp()


def read_ball_frames(path):
    """{half: (N, time, x, y, possession, status)} arrays, one row per ball frame, in file order.

    x and y stay in the provider's centred metres here; build() shifts them to the 0..105 /
    0..68 corner origin the rest of RONDO uses.
    """
    out = {half: [] for half in HALVES}
    section = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "BallStatus" in line:
                if section is None:
                    raise ValueError("{}: ball frame outside a FrameSet".format(path))
                attrs = dict(ATTRS.findall(line))
                out[section].append((
                    int(attrs["N"]), epoch(attrs["T"]),
                    float(attrs["X"]), float(attrs["Y"]),
                    int(attrs["BallPossession"]), int(attrs["BallStatus"])))
            elif "<FrameSet" in line:
                match = FRAMESET.search(line)
                if match:
                    attrs = dict(ATTRS.findall(match.group(1)))
                    section = attrs.get("GameSection")
                    if section not in out:
                        raise ValueError("{}: unknown GameSection {!r}".format(path, section))
    missing = [half for half in HALVES if not out[half]]
    if missing:
        raise ValueError("{}: no ball frames for {}".format(path, missing))
    return {half: np.array(rows, dtype=float) for half, rows in out.items()}


def event_owners(path, home_id, away_id):
    """[(time, code)] every event that names a team, for the independent mapping check.

    Only the Team attribute is used -- the event taxonomy itself is not needed here, and reading
    it would mean taking a position on DFL's event semantics for no gain.
    """
    out = []
    for _, element in ElementTree.iterparse(path, events=("end",)):
        if element.tag != "Event":
            continue
        when = element.get("EventTime")
        for child in element.iter():
            team = child.get("Team")
            if team and when:
                out.append((epoch(when), labels.H if team == home_id else
                            labels.A if team == away_id else labels.X))
                break
        element.clear()
    return [(when, code) for when, code in out if code != labels.X]


def event_agreement(times, codes, owners, tolerance=1.0):
    """Share of events whose team matches the frame label at that moment, and how many counted.

    Only alive frames count: an event timed inside a stoppage says nothing about who was in
    possession. Events more than `tolerance` seconds from any frame are skipped.
    """
    if not owners:
        return 0.0, 0
    when = np.array([t for t, _ in owners])
    want = np.array([c for _, c in owners], dtype=np.int8)
    index = np.clip(np.searchsorted(times, when), 0, len(times) - 1)
    near = np.abs(times[index] - when) <= tolerance
    alive = np.isin(codes[index], (labels.H, labels.A))
    keep = near & alive
    if not keep.any():
        return 0.0, 0
    return float((codes[index][keep] == want[keep]).mean()), int(keep.sum())


def build(match_id, external_dir=None, check_events=True):
    """(codes, truth, meta) for one IDSSE match. Needs the downloaded XMLs on disk."""
    external_dir = external_dir or datasets.EXTERNAL_DIR
    info_path, events_path, positions_path = datasets.require(
        "idsse", match_id, ("matchinfo", "events", "positions"), external_dir)
    info = read_matchinfo(info_path)
    halves = read_ball_frames(positions_path)

    # Halves are concatenated in play order, so frame 0 is the first tracked frame of the match
    # and the second half follows the first with no gap. The provider's own frame numbers are
    # kept in truth["frame_n"] for anyone who needs to go back to the source file.
    rows = np.concatenate([halves[half][np.argsort(halves[half][:, 0])] for half in HALVES])
    number, times, x, y, possession, status = (rows[:, i] for i in range(6))
    pitch_x, pitch_y = info["pitch"]

    codes = np.where(status < 0.5, labels.D,
                     np.where(possession < 1.5, labels.H, labels.A)).astype(np.int8)
    truth = {
        "ball_xy": np.stack([x + pitch_x / 2.0, y + pitch_y / 2.0], axis=1),
        "ball_missing": np.zeros(len(rows), dtype=bool),  # IDSSE tracks the ball every frame
        "half": np.repeat([1, 2], [len(halves[half]) for half in HALVES]).astype(np.int8),
        "frame_n": number.astype(np.int64),
    }

    agreement, n_checked = (0.0, 0)
    if check_events:
        agreement, n_checked = event_agreement(
            times, codes, event_owners(events_path, info["home_team_id"], info["away_team_id"]))
        if n_checked and agreement < MIN_EVENT_AGREEMENT:
            raise ValueError(
                "{}: per-frame possession agrees with the event file on only {:.1%} of {} events."
                " BallPossession 1=home is probably not the mapping in this file -- check before"
                " using these labels.".format(match_id, agreement, n_checked))

    gaps = np.diff(number[truth["half"] == 1]).astype(np.int64)
    meta = dict(info)
    meta.update({
        "dataset": "idsse",
        "fps": datasets.FPS,
        "label_source": "BallStatus + BallPossession from the IDSSE positional XML",
        "event_agreement": round(agreement, 4),
        "event_agreement_n": n_checked,
        "frames_per_half": [int(len(halves[half])) for half in HALVES],
        "max_frame_gap": int(gaps.max()) if len(gaps) else 1,
        "native_ball_missing": 0.0,
    })
    meta.update(labels.summary(codes, datasets.FPS))
    return codes, truth, meta


def write(match_id, external_dir=None, check_events=True):
    """Build one match and save labels.csv + truth.npz + meta.json. Returns (path, meta)."""
    external_dir = external_dir or datasets.EXTERNAL_DIR
    codes, truth, meta = build(match_id, external_dir, check_events)
    folder = labels.match_dir(external_dir, "idsse", match_id)
    return labels.write(folder, codes, truth, meta), meta


def built(match_id, external_dir=None):
    folder = labels.match_dir(external_dir or datasets.EXTERNAL_DIR, "idsse", match_id)
    return os.path.isfile(os.path.join(folder, "labels.csv"))
