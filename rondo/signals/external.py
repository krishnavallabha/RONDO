"""Player tracking of the harness's external matches, in the shape build_evidence takes.

The harness (rondo/harness) keeps one ball track and one label per frame for each external match
(truth.npz + labels.csv). build_evidence also needs the players, so this module reads them from
the same downloaded files and lines them up row for row with the harness's truth.npz.

    players(dataset, match_id) -> (player_xy [N, P, 2] metres from a pitch corner, player_team [P])

Slots are one per tracked person; 0 = home, 1 = away; NaN where a person is not tracked that
frame. Referees and the ball are not slots. The result is cached under data/external/tracking/
(gitignored with the rest of data/), because the IDSSE XML takes about half a minute to scan.
"""

import os
import re

import numpy as np
import pandas as pd

from rondo.harness import datasets, idsse, labels, metrica

FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"')
FRAMESET = re.compile(r"<FrameSet\b([^>]*)>")
ATTRS = re.compile(r'([\w:-]+)="([^"]*)"')


def cache_path(dataset, match_id, external_dir=None):
    return os.path.join(external_dir or datasets.EXTERNAL_DIR, "tracking", dataset, match_id,
                        "players.npz")


def _idsse(match_id, external_dir):
    info_path, positions_path = datasets.require(
        "idsse", match_id, ("matchinfo", "positions"), external_dir)
    info = idsse.read_matchinfo(info_path)
    pitch_x, pitch_y = info["pitch"]
    with np.load(os.path.join(labels.match_dir(external_dir, "idsse", match_id), "truth.npz")) as t:
        half, number = t["half"], t["frame_n"]
    # Row i of the harness is the i-th ball frame, first half then second, each in frame order.
    rows = {section: (np.flatnonzero(half == code), number[half == code])
            for section, code in (("firstHalf", 1), ("secondHalf", 2))}
    team_of = {info["home_team_id"]: 0, info["away_team_id"]: 1}

    slots, teams = {}, []
    columns = []  # (slot, row indices, x, y)
    section, person, frames = None, None, []

    def flush():
        if person is None or not frames:
            return
        n, x, y = (np.array(column) for column in zip(*frames))
        index, frame_numbers = rows[section]
        at = np.searchsorted(frame_numbers, n.astype(np.int64))
        at[at >= len(frame_numbers)] = 0
        ok = frame_numbers[at] == n
        columns.append((slots[person], index[at[ok]], x[ok], y[ok]))

    with open(positions_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if person is not None:
                    match = FRAME.match(line)
                    if match:
                        frames.append((float(match.group(1)), float(match.group(2)),
                                       float(match.group(3))))
            elif "<FrameSet" in line:
                flush()
                frames = []
                attrs = dict(ATTRS.findall(FRAMESET.search(line).group(1)))
                section = attrs.get("GameSection")
                person = None
                team = team_of.get(attrs.get("TeamId"))
                if team is not None and section in rows:
                    # A player seen in both halves is one slot.
                    person = attrs["PersonId"]
                    if person not in slots:
                        slots[person] = len(teams)
                        teams.append(team)
    flush()

    player_xy = np.full((len(half), len(teams), 2), np.nan)
    for slot, index, x, y in columns:
        player_xy[index, slot, 0] = x + pitch_x / 2.0
        player_xy[index, slot, 1] = y + pitch_y / 2.0
    return player_xy, np.array(teams, dtype=np.int8)


def _metrica(match_id, external_dir):
    home_path, away_path = datasets.require(
        "metrica", match_id, ("tracking_home", "tracking_away"), external_dir)
    xy, teams, index = [], [], None
    for team, path in ((0, home_path), (1, away_path)):
        frame = pd.read_csv(path, skiprows=2)
        index = frame["Frame"].to_numpy(dtype=np.int64) - 1
        values = frame.iloc[:, 3:-2].to_numpy(dtype=float)  # (x, y) pairs, ball excluded
        xy.append(values.reshape(len(frame), -1, 2) * metrica.PITCH)
        teams += [team] * xy[-1].shape[1]
    n_frames = int(index.max()) + 1
    player_xy = np.full((n_frames, len(teams), 2), np.nan)
    player_xy[index] = np.concatenate(xy, axis=1)
    return player_xy, np.array(teams, dtype=np.int8)


def players(dataset, match_id, external_dir=None, refresh=False):
    """(player_xy, player_team) for one external match, row-aligned with its truth.npz."""
    external_dir = external_dir or datasets.EXTERNAL_DIR
    path = cache_path(dataset, match_id, external_dir)
    if os.path.isfile(path) and not refresh:
        with np.load(path) as data:
            return data["player_xy"], data["player_team"]
    player_xy, team = (_idsse if dataset == "idsse" else _metrica)(match_id, external_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, player_xy=player_xy, player_team=team)
    return player_xy, team
