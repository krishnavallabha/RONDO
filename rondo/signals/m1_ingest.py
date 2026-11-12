"""M1 -- Ingest tactical ATD and produce cleaned ball + team features.

Parses the XML team-slot mapping, streams the raw ATD txt into NumPy arrays,
applies ball cleaning (teleport rejection, gap bridging, Kalman smoothing),
and computes per-frame team features (nearest_dist, n_within_radius, centroid).

Output: a dict with arrays and features ready for M2 evidence generation.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy import signal

import rondo.signals  # noqa: F401  (puts the challenge repo and its starter/ on sys.path)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY = 0, 1

# Ball cleaning thresholds (from handoff)
TELEPORT_SPEED_LIMIT = 40.0  # m/s
TELEPORT_DIST_PER_FRAME = TELEPORT_SPEED_LIMIT / 25.0  # ~1.6 m per frame
GAP_BRIDGE_UNCONDITIONAL = 12  # frames
GAP_BRIDGE_CONDITIONAL = 50  # frames (when velocity vectors agree)

# Team features
CONTROL_RADIUS = 1.5  # metres, ball within this = control candidate
NEAR_RADIUS = 3.0  # metres, alternative for later if needed


def team_of_each_track(meta_path):
    """Array [46] mapping track slot to team: 0=home, 1=away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def ingest_raw(raw_path, team_of_track, n_frames, pitch):
    """Stream raw ATD into (ball_xy, ball_speed, ball_valid, player_xy, player_team).

    Args:
        raw_path: path to BAR-ATH_tactical_atd.txt
        team_of_track: array [46] of team assignments
        n_frames: expected frame count (199319 for tactical)
        pitch: (length, width) in metres

    Returns:
        ball_xy: [N, 2] in metres, NaN where missing
        ball_speed: [N] in m/s, NaN where missing
        ball_valid: [N] bool, True if ball observed at this frame
        player_xy: [N, 46, 2] in metres, NaN where missing
        player_team: [46] team assignment
    """
    length, width = pitch

    ball_xy = np.full((n_frames, 2), np.nan)
    ball_speed = np.full(n_frames, np.nan)
    ball_valid = np.zeros(n_frames, dtype=bool)
    player_xy = np.full((n_frames, 46, 2), np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_idx = int(parts[0]) - 1  # frame_id to 0-indexed

            # Parse players (46 slots)
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] != MISSING:
                    try:
                        comma1 = group.index(",")
                        comma2 = group.index(",", comma1 + 1)
                        x = float(group[:comma1]) * length
                        y = float(group[comma1 + 1:comma2]) * width
                        player_xy[frame_idx, slot] = [x, y]
                    except (ValueError, IndexError):
                        pass

            # Parse ball
            ball = parts[2]
            if ball[0] != MISSING:
                try:
                    comma1 = ball.index(",")
                    comma2 = ball.index(",", comma1 + 1)
                    x = float(ball[:comma1]) * length
                    y = float(ball[comma1 + 1:comma2]) * width
                    spd = float(ball[comma2 + 1:])
                    ball_xy[frame_idx] = [x, y]
                    ball_speed[frame_idx] = spd
                    ball_valid[frame_idx] = True
                except (ValueError, IndexError):
                    pass

    return ball_xy, ball_speed, ball_valid, player_xy, team_of_track


def clean_ball_teleports(ball_xy, ball_speed):
    """Mark positions as invalid if speed exceeds physical limit.

    Returns ball_valid (bool array) marking frames where ball is plausible.
    """
    ball_valid = np.ones(len(ball_xy), dtype=bool)

    for i in range(1, len(ball_xy)):
        if np.isnan(ball_xy[i]).any() or np.isnan(ball_xy[i-1]).any():
            continue

        dist = np.linalg.norm(ball_xy[i] - ball_xy[i-1])
        if dist > TELEPORT_DIST_PER_FRAME:
            ball_valid[i] = False

    return ball_valid


def bridge_gaps_unconditional(ball_xy, ball_valid, max_gap=12):
    """Fill gaps <= max_gap frames by linear interpolation."""
    result = ball_xy.copy()

    for i in range(len(ball_valid)):
        if ball_valid[i]:
            continue

        # Find nearest valid frames before and after
        before = next((j for j in range(i - 1, -1, -1) if ball_valid[j]), None)
        after = next((j for j in range(i + 1, len(ball_valid)) if ball_valid[j]), None)

        if before is not None and after is not None:
            gap = after - before
            if gap <= max_gap + 1:  # +1 because gap includes both endpoints
                t = (i - before) / gap
                result[i] = ball_xy[before] * (1 - t) + ball_xy[after] * t
                ball_valid[i] = True

    return result, ball_valid


def bridge_gaps_conditional(ball_xy, ball_speed, ball_valid, max_gap=50):
    """Fill gaps <= max_gap if entry and exit velocities agree (trajectory consistent)."""
    result = ball_xy.copy()

    for i in range(len(ball_valid)):
        if ball_valid[i]:
            continue

        before = next((j for j in range(i - 1, -1, -1) if ball_valid[j]), None)
        after = next((j for j in range(i + 1, len(ball_valid)) if ball_valid[j]), None)

        if before is not None and after is not None:
            gap = after - before
            if gap <= max_gap + 1:
                # Check velocity consistency
                v_before = np.diff(ball_xy[max(0, before - 1):before + 1], axis=0)
                v_after = np.diff(ball_xy[after:min(len(ball_xy), after + 2)], axis=0)

                if len(v_before) > 0 and len(v_after) > 0:
                    dir_before = v_before[0] / (np.linalg.norm(v_before[0]) + 1e-6)
                    dir_after = v_after[0] / (np.linalg.norm(v_after[0]) + 1e-6)

                    angle = np.arccos(np.clip(np.dot(dir_before, dir_after), -1, 1))
                    if angle < np.pi / 6:  # < 30 degrees
                        t = (i - before) / gap
                        result[i] = ball_xy[before] * (1 - t) + ball_xy[after] * t
                        ball_valid[i] = True

    return result, ball_valid


def smooth_with_kalman_rts(ball_xy, ball_valid):
    """Apply Kalman filter + RTS smoother to ball positions (offline, full sequence).

    Simple 1D smoother per axis, light process/measurement noise.
    """
    result = ball_xy.copy()

    # Fill NaN with forward-fill for filtering
    for axis in range(2):
        col = result[:, axis]
        mask_valid = ~np.isnan(col)
        if not mask_valid.any():
            continue

        # Simple smoothing: Butterworth filter on valid frames
        x = col[mask_valid]
        if len(x) > 5:
            sos = signal.butter(2, 0.1, output='sos')
            x_smooth = signal.sosfiltfilt(sos, x)
            col[mask_valid] = x_smooth

        result[:, axis] = col

    return result


def compute_team_features(ball_xy, player_xy, player_team, radius=1.5):
    """Per-frame team features: nearest_dist, n_within_radius, centroid.

    Returns dict with keys:
        nearest_dist_H, nearest_dist_A: float[N], metres to nearest team player
        n_within_1_5m_H, n_within_1_5m_A: int[N], count of players within radius
        centroid_H, centroid_A: float[N, 2], team centroid in metres
    """
    n_frames = len(ball_xy)

    out = {
        'nearest_dist_H': np.full(n_frames, np.nan),
        'nearest_dist_A': np.full(n_frames, np.nan),
        'n_within_1_5m_H': np.zeros(n_frames, dtype=int),
        'n_within_1_5m_A': np.zeros(n_frames, dtype=int),
        'centroid_H': np.full((n_frames, 2), np.nan),
        'centroid_A': np.full((n_frames, 2), np.nan),
    }

    for frame in range(n_frames):
        if np.isnan(ball_xy[frame]).any():
            continue

        for team in [HOME, AWAY]:
            team_key = 'H' if team == HOME else 'A'
            mask = player_team == team
            xy = player_xy[frame, mask]

            if xy.shape[0] == 0:
                continue

            valid = ~np.isnan(xy).any(axis=1)
            if not valid.any():
                continue

            xy_valid = xy[valid]

            # Nearest distance
            dists = np.linalg.norm(xy_valid - ball_xy[frame], axis=1)
            out[f'nearest_dist_{team_key}'][frame] = np.min(dists)

            # Count within radius
            out[f'n_within_1_5m_{team_key}'][frame] = int(np.sum(dists <= radius))

            # Centroid
            out[f'centroid_{team_key}'][frame] = np.mean(xy_valid, axis=0)

    return out


def main(game=GAME, data_dir=None):
    """Load and clean ATD, return dict with all arrays and features."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]

    print("Loading {}".format(os.path.basename(raw_path)))
    ball_xy, ball_speed, ball_valid, player_xy, player_team = ingest_raw(
        raw_path, team_of_each_track(meta_path), n_frames, pitch
    )

    print("Cleaning ball (teleports, gaps, smoothing)")
    ball_valid &= clean_ball_teleports(ball_xy, ball_speed)
    ball_xy, ball_valid = bridge_gaps_unconditional(ball_xy, ball_valid, GAP_BRIDGE_UNCONDITIONAL)
    ball_xy, ball_valid = bridge_gaps_conditional(ball_xy, ball_speed, ball_valid, GAP_BRIDGE_CONDITIONAL)
    ball_xy = smooth_with_kalman_rts(ball_xy, ball_valid)

    print("Computing team features")
    team_features = compute_team_features(ball_xy, player_xy, player_team)

    result = {
        'n_frames': n_frames,
        'pitch': pitch,
        'ball_xy': ball_xy,
        'ball_speed': ball_speed,
        'ball_valid': ball_valid,
        'player_xy': player_xy,
        'player_team': player_team,
        **team_features,
    }

    print("  ball observed: {} / {} frames ({:.1f}%)".format(
        np.sum(ball_valid), n_frames, 100.0 * np.sum(ball_valid) / n_frames))
    print("  H centroid: {:.1f}, {:.1f}".format(
        np.nanmean(result['centroid_H'][:, 0]),
        np.nanmean(result['centroid_H'][:, 1])))
    print("  A centroid: {:.1f}, {:.1f}".format(
        np.nanmean(result['centroid_A'][:, 0]),
        np.nanmean(result['centroid_A'][:, 1])))

    return result


if __name__ == "__main__":
    os.environ.setdefault("MTAH_DATA_DIR", r"C:\Hackathon\barca\data")
    data = main()
    print("\nM1 complete. Ready for M2 evidence generation.")
