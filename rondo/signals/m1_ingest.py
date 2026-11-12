"""M1 -- Ingest tactical ATD and produce cleaned ball + team features.

Parses the XML team-slot mapping, streams the raw ATD txt into NumPy arrays,
applies ball cleaning (out-of-pitch and teleport rejection, gap bridging, zero-phase smoothing),
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

FPS = 25

# Ball cleaning thresholds (from handoff)
TELEPORT_SPEED_LIMIT = 40.0  # m/s
TELEPORT_DIST_PER_FRAME = TELEPORT_SPEED_LIMIT / FPS  # ~1.6 m per frame
GAP_BRIDGE_UNCONDITIONAL = 12  # frames
GAP_BRIDGE_CONDITIONAL = 50  # frames (when velocity vectors agree)

# Ball cleaning details
POSITION_MARGIN = 10.0        # m beyond a pitch line; further out is tracking junk, not play
EDGE_MIN_FRAMES = 5           # a leading/trailing segment shorter than this next to a jump is dropped
VELOCITY_FRAMES = 3           # frames averaged to get the velocity at a gap edge
STATIONARY_STEP = 0.05        # m/frame (1.25 m/s): below this the ball counts as still
VELOCITY_ANGLE_DEG = 30.0     # entry/exit directions must agree within this
VELOCITY_SPEED_RATIO = 3.0    # entry/exit speeds must agree within this factor (ball slows in flight)
SMOOTH_CUTOFF = 0.1           # Butterworth cutoff, fraction of Nyquist (1.25 Hz at 25 fps)
SMOOTH_MIN_RUN = 16           # frames; shorter observed runs are not smoothed

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


def _runs(mask):
    """[(start, stop)) of each maximal run of True in `mask`."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def reject_out_of_pitch(ball_xy, pitch, margin=POSITION_MARGIN):
    """NaN the positions further than `margin` metres outside the pitch. Returns (xy, n_rejected)."""
    length, width = pitch
    xy = ball_xy.copy()
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        bad = (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)
    xy[bad] = np.nan
    return xy, int(bad.sum())


def reject_teleports(ball_xy, max_step=TELEPORT_DIST_PER_FRAME, min_edge=EDGE_MIN_FRAMES):
    """Remove impossible jumps (more than `max_step` metres per elapsed frame). Returns (xy, n_removed).

    The track is cut into segments at every impossible jump. A segment is an excursion, and is
    dropped, when the track on both sides of it reconnects without it and it is no longer than
    either neighbour: the tracker briefly followed something else and came back. Segments that
    do not reconnect are a real relocation, such as the tracker re-locking on the ball after a
    miss, so they are kept as separate stretches with a break at the jump (the first frame after
    it is blanked) so that nothing later smooths or differentiates across it. A leading or
    trailing segment shorter than `min_edge` frames next to a jump is dropped as unverifiable.
    """
    xy = ball_xy.copy()
    idx = np.flatnonzero(~np.isnan(ball_xy).any(axis=1))
    if len(idx) < 2:
        return xy, 0
    pts = ball_xy[idx]
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    jump = steps > max_step * np.diff(idx)
    bounds = np.concatenate(([0], np.flatnonzero(jump) + 1, [len(idx)]))
    kept = [np.arange(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]

    def reachable(a, b):
        return np.linalg.norm(pts[b] - pts[a]) <= max_step * (idx[b] - idx[a])

    merged = []
    for seg in kept:
        merged.append(seg)
        while len(merged) >= 3:
            s0, s1, s2 = merged[-3:]
            if len(s1) <= min(len(s0), len(s2)) and reachable(s0[-1], s2[0]):
                merged[-3:] = [np.concatenate([s0, s2])]
            else:
                break
    if len(merged) >= 2 and len(merged[0]) < min_edge:
        merged = merged[1:]
    if len(merged) >= 2 and len(merged[-1]) < min_edge:
        merged = merged[:-1]

    keep = np.zeros(len(idx), dtype=bool)
    for seg in merged:
        keep[seg] = True
    xy[idx[~keep]] = np.nan
    n_removed = int((~keep).sum())
    for prev, cur in zip(merged[:-1], merged[1:]):
        if idx[cur[0]] - idx[prev[-1]] == 1:
            xy[idx[cur[0]]] = np.nan
            n_removed += 1
    return xy, n_removed


def _edge_velocity(xy, index, direction, k=VELOCITY_FRAMES):
    """Mean velocity in metres per frame over up to `k` frames next to a gap edge, or None.
    direction=-1: window ends at `index` (entry); direction=+1: window starts at `index` (exit).
    """
    for span in range(k, 0, -1):
        other = index + direction * span
        if 0 <= other < len(xy) and not np.isnan(xy[other]).any():
            delta = xy[index] - xy[other] if direction < 0 else xy[other] - xy[index]
            return delta / span
    return None


def _velocities_agree(v_in, v_out):
    """Entry and exit velocities describe one continuous trajectory: same direction, similar speed."""
    if v_in is None or v_out is None:
        return False
    s_in, s_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
    if s_in < STATIONARY_STEP and s_out < STATIONARY_STEP:
        return True  # the ball sat still on both sides of the gap
    if s_in < STATIONARY_STEP or s_out < STATIONARY_STEP:
        return False
    cos = np.clip(np.dot(v_in, v_out) / (s_in * s_out), -1.0, 1.0)
    if np.degrees(np.arccos(cos)) > VELOCITY_ANGLE_DEG:
        return False
    return 1.0 / VELOCITY_SPEED_RATIO <= s_out / s_in <= VELOCITY_SPEED_RATIO


def bridge_gaps(ball_xy, max_unconditional=GAP_BRIDGE_UNCONDITIONAL,
                max_conditional=GAP_BRIDGE_CONDITIONAL):
    """Linear interpolation across short interior gaps. Returns (xy, n_unconditional, n_conditional)
    counted in frames.

    Gaps up to `max_unconditional` frames are always bridged. Gaps up to `max_conditional` are
    bridged only when the velocity entering the gap agrees with the one leaving it. Either way the
    implied speed must stay under the teleport limit. Gaps at the very start or end are left alone.
    """
    xy = ball_xy.copy()
    n_uncond = n_cond = 0
    for start, stop in _runs(np.isnan(ball_xy).any(axis=1)):
        length = stop - start
        if start == 0 or stop >= len(xy) or length > max_conditional:
            continue
        before, after = start - 1, stop
        if np.linalg.norm(xy[after] - xy[before]) > TELEPORT_DIST_PER_FRAME * (length + 1):
            continue
        if length > max_unconditional:
            v_in = _edge_velocity(xy, before, -1)
            v_out = _edge_velocity(xy, after, +1)
            if not _velocities_agree(v_in, v_out):
                continue
            n_cond += length
        else:
            n_uncond += length
        t = (np.arange(start, stop) - before) / (after - before)
        xy[start:stop] = xy[before] + t[:, None] * (xy[after] - xy[before])
    return xy, n_uncond, n_cond


def smooth_runs(ball_xy, cutoff=SMOOTH_CUTOFF, min_len=SMOOTH_MIN_RUN):
    """Zero-phase 2nd-order Butterworth low-pass, applied separately to each contiguous run of
    observed frames so nothing is smoothed across a gap. Runs shorter than `min_len` are kept raw.
    """
    sos = signal.butter(2, cutoff, output="sos")
    xy = ball_xy.copy()
    for start, stop in _runs(~np.isnan(ball_xy).any(axis=1)):
        if stop - start >= min_len:
            for axis in range(2):
                xy[start:stop, axis] = signal.sosfiltfilt(sos, ball_xy[start:stop, axis])
    return xy


def ball_speed_from_positions(ball_xy, fps=FPS):
    """Ball speed in m/s from consecutive cleaned positions. NaN where either frame is missing."""
    speed = np.full(len(ball_xy), np.nan)
    step = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1)
    speed[1:] = step * fps
    return speed


def clean_ball(ball_xy, pitch):
    """Full ball cleaning. Returns (xy, stats) where xy is NaN wherever the ball is not trusted."""
    xy, n_oob = reject_out_of_pitch(ball_xy, pitch)
    xy, n_tele = reject_teleports(xy)
    xy, n_uncond, n_cond = bridge_gaps(xy)
    xy = smooth_runs(xy)
    stats = {"out_of_pitch": n_oob, "teleports": n_tele,
             "bridged_short": n_uncond, "bridged_long": n_cond}
    return xy, stats


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
    ball_xy_raw, ball_speed_raw, ball_observed, player_xy, player_team = ingest_raw(
        raw_path, team_of_each_track(meta_path), n_frames, pitch
    )

    print("Cleaning ball (out-of-pitch, teleports, gaps, smoothing)")
    ball_xy, clean_stats = clean_ball(ball_xy_raw, pitch)
    ball_valid = ~np.isnan(ball_xy).any(axis=1)
    ball_speed = ball_speed_from_positions(ball_xy)

    print("Computing team features")
    team_features = compute_team_features(ball_xy, player_xy, player_team)

    result = {
        'n_frames': n_frames,
        'pitch': pitch,
        'ball_xy': ball_xy,
        'ball_speed': ball_speed,        # m/s, from the cleaned positions
        'ball_speed_raw': ball_speed_raw,  # the ATD's own speed field, uncleaned
        'ball_valid': ball_valid,
        'player_xy': player_xy,
        'player_team': player_team,
        'clean_stats': clean_stats,
        **team_features,
    }

    print("  ball in raw ATD: {} / {} frames ({:.1f}%)".format(
        int(ball_observed.sum()), n_frames, 100.0 * ball_observed.sum() / n_frames))
    print("  rejected: {} out of pitch, {} teleports; bridged: {} short-gap, {} long-gap frames".format(
        clean_stats["out_of_pitch"], clean_stats["teleports"],
        clean_stats["bridged_short"], clean_stats["bridged_long"]))
    print("  ball usable after cleaning: {} / {} frames ({:.1f}%)".format(
        int(ball_valid.sum()), n_frames, 100.0 * ball_valid.sum() / n_frames))
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
