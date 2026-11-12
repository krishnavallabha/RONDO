"""M2 -- Evidence Generation (per-frame emission signals).

Loads M1 output (cleaned ball + player positions) and Smart Tagging,
generates per-frame evidence signals for the M3 decoder. Offline / non-causal.

Signals: touch (H/A), control (sustained proximity), duel zone, dead ball,
restart owner, Smart Tagging priors.
"""

import os
import sys
import numpy as np
from typing import Tuple, Dict

import rondo.signals  # noqa: F401  (puts the challenge repo and its starter/ on sys.path)
from games import GAME, GAMES, source_frames  # noqa: E402
from load_smart_tagging import load as load_smart_tagging  # noqa: E402

from rondo.signals.m1_ingest import main as ingest_atd  # noqa: E402

# Params
FPS = 25
CONTROL_RADIUS = 1.5            # m
MIN_PLAYER_DIST = 1.5           # m, touch proximity
VEL_WINDOW = 6                  # frames each side; M1's 1.25 Hz low-pass spreads a kick over ~0.5 s
TOUCH_IMPULSE = 4.0             # m/s, |v_after - v_before| (kick, trap, block)
TOUCH_DIRECTION_THRESHOLD = 30  # degrees
TOUCH_DIRECTION_MIN_SPEED = 3.0 # m/s
TOUCH_NMS = 3                   # frames, keep one touch per local impulse peak
CONTROL_PERSISTENCE_K = 5       # frames (0.2 s)
CONTROL_GAP_CLOSE = 2           # frames of dropout tolerated inside a control run
CONTROL_MAX_BALL_SPEED = 10.0   # m/s, faster than this the ball is travelling, not carried
BALL_STATIONARY = 0.3           # m/s
STATIONARY_MIN_FRAMES = 12      # 0.5 s
PLAYER_SLOW = 1.5               # m/s, mean player speed during a stoppage
BALL_LIVE_SPEED = 3.0           # m/s, moving ball inside the pitch -> evidence of live
OOB_MARGIN = 0.5                # m beyond pitch bounds
BOUNDARY_NEAR = 2.0             # m from a pitch line
MISSING_BALL_AT_BOUNDARY_GAP = 50  # frames
RESTART_PAD = 10                # frames each side of a localised restart frame
SMART_TAG_DEAD_SHORT = 0.8      # prior on short restart tags (kickoffs, ~5 s)
SMART_TAG_DEAD_LONG = 0.6       # most SET PIECES tags are a fixed 875-frame (35 s) window
SHORT_TAG_FRAMES = 250
# Published kickoff frames (DATA.md / FAQ.md). 1st-half owner A: public GT frame 11185 is 'A'
# (also visible on video); 2nd half is the other team by the Laws of the Game.
KICKOFFS = {"1st Half": (11185, 1), "2nd Half": (113742, 0)}

HOME, AWAY = 0, 1
TEAM_ALIASES = {HOME: ("barcelona", "barca", "barça"),
                AWAY: ("athletic", "atletic", "bilbao")}


# ---------------------------------------------------------------- kinematics

def ball_velocity(ball_xy: np.ndarray, w: int = VEL_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """Backward and forward ball velocity [N,2] in m/s over w frames (NaN at gaps/edges)."""
    n = len(ball_xy)
    v_before = np.full((n, 2), np.nan)
    v_after = np.full((n, 2), np.nan)
    v_before[w:] = (ball_xy[w:] - ball_xy[:-w]) * FPS / w
    v_after[:-w] = (ball_xy[w:] - ball_xy[:-w]) * FPS / w
    return v_before, v_after


def ball_speed_mps(ball_xy: np.ndarray) -> np.ndarray:
    """Central-difference ball speed in m/s from cleaned positions; NaN where undefined."""
    sp = np.full(len(ball_xy), np.nan)
    d = np.linalg.norm(ball_xy[2:] - ball_xy[:-2], axis=1) * FPS / 2.0
    sp[1:-1] = d
    return sp


def angle_diff(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """Smallest angle difference in degrees (inputs in radians, any range)."""
    diff = np.abs(angle1 - angle2) % (2 * np.pi)
    return np.minimum(diff, 2 * np.pi - diff) * 180 / np.pi


def team_nearest(ball_xy: np.ndarray, player_xy: np.ndarray,
                 player_team: np.ndarray) -> np.ndarray:
    """[N,2] distance (m) from ball to nearest player of H (col 0) and A (col 1); NaN if none."""
    n = len(ball_xy)
    out = np.full((n, 2), np.nan)
    for team in (HOME, AWAY):
        xy = player_xy[:, player_team == team, :]
        d = np.linalg.norm(xy - ball_xy[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        m = d.min(axis=1) if d.shape[1] else np.full(n, np.inf)
        m[np.isinf(m)] = np.nan
        out[:, team] = m
    return out


def _runs(mask: np.ndarray):
    """(start, end) half-open runs of True."""
    m = np.concatenate(([False], mask.astype(bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Fill False runs of length <= gap that sit between two True runs."""
    out = mask.copy()
    for s, e in _runs(~mask):
        if s > 0 and e < len(mask) and e - s <= gap:
            out[s:e] = True
    return out


# ---------------------------------------------------------------- signals

def detect_touches(ball_xy: np.ndarray, nearest: np.ndarray,
                   impulse_threshold: float = TOUCH_IMPULSE,
                   direction_threshold: float = TOUCH_DIRECTION_THRESHOLD,
                   min_player_dist: float = MIN_PLAYER_DIST) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-level touch events by H and A.

    Ball event at f if |v_after - v_before| > impulse_threshold (kick / trap / block), OR
    direction turns by > direction_threshold deg while speed > TOUCH_DIRECTION_MIN_SPEED.
    Events are thinned to local impulse peaks (+-TOUCH_NMS), then attributed to every team
    with a player within min_player_dist (tolerating +-1 frame). NaN ball -> no touch.
    """
    n = len(ball_xy)
    vb, va = ball_velocity(ball_xy)
    impulse = np.linalg.norm(va - vb, axis=1)
    sb, sa = np.linalg.norm(vb, axis=1), np.linalg.norm(va, axis=1)
    with np.errstate(invalid="ignore"):
        turn = angle_diff(np.arctan2(va[:, 1], va[:, 0]), np.arctan2(vb[:, 1], vb[:, 0]))
        deflect = (turn > direction_threshold) & (np.minimum(sb, sa) > TOUCH_DIRECTION_MIN_SPEED)
        event = (impulse > impulse_threshold) | deflect

    # Non-maximum suppression on impulse so one physical touch = one frame
    imp = np.where(np.isnan(impulse), -np.inf, impulse)
    pad = np.pad(imp, TOUCH_NMS, constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * TOUCH_NMS + 1)
    peak = imp >= win.max(axis=1)
    event &= peak

    # Proximity, min over f-1..f+1 to absorb residual timing error
    nd = np.where(np.isnan(nearest), np.inf, nearest)
    nd_tol = nd.copy()
    nd_tol[1:] = np.minimum(nd_tol[1:], nd[:-1])
    nd_tol[:-1] = np.minimum(nd_tol[:-1], nd[1:])
    touch_H = event & (nd_tol[:, HOME] <= min_player_dist)
    touch_A = event & (nd_tol[:, AWAY] <= min_player_dist)
    return touch_H, touch_A


def filter_control_persistence(cand_H: np.ndarray, cand_A: np.ndarray,
                               k: int = CONTROL_PERSISTENCE_K,
                               gap: int = CONTROL_GAP_CLOSE) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only runs of >= k consecutive candidate frames (after closing <= gap dropouts)."""
    def filt(c):
        c = _close_gaps(c.astype(bool), gap)
        out = np.zeros(len(c), dtype=bool)
        for s, e in _runs(c):
            if e - s >= k:
                out[s:e] = True
        return out
    return filt(cand_H), filt(cand_A)


def control_candidates(nearest: np.ndarray, speed: np.ndarray,
                       radius: float = CONTROL_RADIUS) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame 'ball is at a team's feet': that team's nearest player is within radius,
    closer than the opponent's, and the ball is slow enough to be carried."""
    nd = np.where(np.isnan(nearest), np.inf, nearest)
    slow = np.nan_to_num(speed, nan=np.inf) < CONTROL_MAX_BALL_SPEED
    cH = (nd[:, HOME] <= radius) & (nd[:, HOME] < nd[:, AWAY]) & slow
    cA = (nd[:, AWAY] <= radius) & (nd[:, AWAY] < nd[:, HOME]) & slow
    return cH, cA


def detect_duel_zone(nearest: np.ndarray, radius: float = CONTROL_RADIUS) -> np.ndarray:
    """Both teams have a player within radius of the ball (False where ball missing)."""
    nd = np.where(np.isnan(nearest), np.inf, nearest)
    return (nd[:, HOME] <= radius) & (nd[:, AWAY] <= radius)


def detect_dead_ball(ball_xy: np.ndarray, speed: np.ndarray, player_xy: np.ndarray,
                     pitch: Tuple[float, float]) -> np.ndarray:
    """Soft probability that the ball is dead, float [N] in [0,1].

    1.0  ball beyond pitch + OOB_MARGIN
    0.9  ball missing and its last known position was out of bounds
    0.8  ball stationary (< BALL_STATIONARY m/s for >= STATIONARY_MIN_FRAMES) and players slow
    0.7  ball missing > MISSING_BALL_AT_BOUNDARY_GAP frames after last seen within BOUNDARY_NEAR
         of a line
    0.3  ball moving > BALL_LIVE_SPEED inside the pitch (evidence of live play)
    0.5  otherwise (unknown)
    """
    n = len(ball_xy)
    length, width = pitch
    p = np.full(n, 0.5)
    present = ~np.isnan(ball_xy).any(axis=1)
    x, y = ball_xy[:, 0], ball_xy[:, 1]

    with np.errstate(invalid="ignore"):
        oob = present & ((x < -OOB_MARGIN) | (x > length + OOB_MARGIN) |
                         (y < -OOB_MARGIN) | (y > width + OOB_MARGIN))
        near_line = present & ((x < BOUNDARY_NEAR) | (x > length - BOUNDARY_NEAR) |
                               (y < BOUNDARY_NEAR) | (y > width - BOUNDARY_NEAR))
        live = present & ~oob & (speed > BALL_LIVE_SPEED)
    p[live] = 0.3

    # Stationary ball, sustained, with slow players
    step = np.linalg.norm(np.diff(player_xy, axis=0), axis=2) * FPS  # [N-1, 46] m/s
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pmean = np.concatenate(([np.nan], np.nanmean(step, axis=1)))
        still = present & (np.nan_to_num(speed, nan=np.inf) < BALL_STATIONARY)
    still_long = np.zeros(n, dtype=bool)
    for s, e in _runs(still):
        if e - s >= STATIONARY_MIN_FRAMES:
            still_long[s:e] = True
    p[still_long & (np.nan_to_num(pmean, nan=np.inf) < PLAYER_SLOW)] = 0.8

    # Missing ball: propagate last known position forward (vectorised forward-fill)
    idx = np.where(present, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    miss = ~present & (last >= 0)
    gap_len = np.arange(n) - last
    lv = last[miss]
    p_miss = np.full(miss.sum(), 0.5)
    p_miss[near_line[lv] & (gap_len[miss] > MISSING_BALL_AT_BOUNDARY_GAP)] = 0.7
    p_miss[oob[lv]] = 0.9
    p[miss] = p_miss

    p[oob] = 1.0
    return p


# ---------------------------------------------------------------- smart tagging

def infer_team_mapping(smart_tagging: list) -> Dict[str, int]:
    """Team tag value -> HOME/AWAY. Barcelona = H, Athletic ('Atletic Club' in the file) = A."""
    team_map = {}
    for phase in smart_tagging:
        name = (phase['tags'].get('Team') or '').strip()
        low = name.lower()
        for code, aliases in TEAM_ALIASES.items():
            if any(a in low for a in aliases):
                team_map[name] = code
    return team_map


def load_smart_tagging_phases(smart_tagging: list, n_frames: int,
                              team_map: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """smart_tag_team int8[N] (-1/0/1) and smart_tag_dead float[N] (0.8 on SET PIECES/GOALS).

    Phases overlap; set pieces are written last so their team wins over a phase-of-play tag.
    """
    smart_tag_team = np.full(n_frames, -1, dtype=np.int8)
    smart_tag_dead = np.full(n_frames, 0.5)

    def is_restart(ph):
        c = (ph['code'] or '').strip().lower()
        return 'set pieces' in c or 'goals' in c

    for phase in sorted(smart_tagging, key=is_restart):
        s, e = max(0, phase['start_frame']), min(n_frames, phase['end_frame'])
        if e <= s:
            continue
        team = team_map.get((phase['tags'].get('Team') or '').strip())
        if team is not None:
            smart_tag_team[s:e] = team
        if is_restart(phase):
            smart_tag_dead[s:e] = SMART_TAG_DEAD_SHORT if e - s <= SHORT_TAG_FRAMES \
                else SMART_TAG_DEAD_LONG
    return smart_tag_team, smart_tag_dead


def localise_restart(s: int, e: int, dead_ball, touch_H, touch_A, team=None):
    """(frame, team) of the restart inside a tag window [s,e).

    Restart = first touch after the last dead stretch (dead_ball >= 0.7) in the window; if there
    is no dead stretch, the first touch (by `team` when known). Returns (None, -1) if no touch.
    """
    dead = dead_ball[s:e] >= 0.7
    after = s + (int(np.flatnonzero(dead)[-1]) + 1 if dead.any() else 0)
    for f in range(after, e):
        if team is None and touch_H[f] != touch_A[f]:
            return f, (HOME if touch_H[f] else AWAY)
        if team is not None and (touch_H[f] if team == HOME else touch_A[f]):
            return f, team
    if dead.any() and team is not None:
        return after, team        # ball left the dead stretch; no clean touch seen
    return None, -1


def infer_restart_team(smart_tagging: list, team_map: Dict[str, int], n_frames: int,
                       dead_ball, touch_H, touch_A) -> Tuple[np.ndarray, Dict[str, int]]:
    """restart_team int8[N]: who takes each restart, on +-RESTART_PAD frames around it.

    - Kick Off (Start): published kickoff frames and owners (KICKOFFS).
    - Corner / goal kick / free kick: Team tag = team performing the set piece.
    - Kick Off (After Goal): team that conceded = opposite of the preceding GOALS tag.
    - Throw-ins (Team 'N/A'): team of the first touch after the dead stretch, from tracking.
    The restart frame inside each (mostly 35 s) tag window is localised from tracking.
    """
    restart = np.full(n_frames, -1, dtype=np.int8)
    stats = {"kickoff": 0, "tag": 0, "after_goal": 0, "tracking": 0, "unlocated": 0}
    goals = sorted((p['start_frame'], team_map.get((p['tags'].get('Team') or '').strip()))
                   for p in smart_tagging if (p['code'] or '').strip().lower() == 'goals')

    def mark(f, team):
        restart[max(0, f - RESTART_PAD):min(n_frames, f + RESTART_PAD + 1)] = team

    for f, team in KICKOFFS.values():
        mark(f, team)
        stats["kickoff"] += 1

    for p in smart_tagging:
        if (p['code'] or '').strip().lower() != 'set pieces':
            continue
        typ = (p['tags'].get('Type') or '').strip().lower()
        if 'kick off (start)' in typ:
            continue
        s, e = max(0, p['start_frame']), min(n_frames, p['end_frame'])
        team = team_map.get((p['tags'].get('Team') or '').strip())
        src = "tag"
        if team is None and 'after goal' in typ:
            prev = [t for g, t in goals if g <= s and t is not None]
            team, src = ((1 - prev[-1]), "after_goal") if prev else (None, src)
        f, found = localise_restart(s, e, dead_ball, touch_H, touch_A, team)
        if f is None:
            stats["unlocated"] += 1
            continue
        stats[src if team is not None else "tracking"] += 1
        mark(f, found)
    return restart, stats


# ---------------------------------------------------------------- pipeline

def main(game: str = GAME, data_dir: str = None, m1_data: dict = None) -> Dict[str, np.ndarray]:
    """Full M2 pipeline. Returns the M2->M3 evidence dict:
        ball_xy float[N,2], ball_missing bool[N], touch_H/touch_A bool[N],
        control_H/control_A bool[N], duel_zone bool[N], dead_ball float[N] in [0,1],
        restart_team int8[N] (-1/0/1), smart_tag_dead float[N], smart_tag_team int8[N]
    """
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]

    if m1_data is None:
        print("M2: Loading M1 output")
        m1_data = ingest_atd(game, data_dir)
    ball_xy = m1_data['ball_xy']
    player_xy = m1_data['player_xy']
    player_team = m1_data['player_team']

    print("M2: Kinematics")
    speed = ball_speed_mps(ball_xy)  # recomputed from cleaned positions, m/s
    nearest = team_nearest(ball_xy, player_xy, player_team)

    print("M2: Loading Smart Tagging")
    smart_tagging = load_smart_tagging(game, data_dir)
    team_map = infer_team_mapping(smart_tagging)
    if set(team_map.values()) != {HOME, AWAY}:
        print("  WARNING: Team tags not mapped for both sides: {}".format(team_map))

    print("M2: Touches / control / duel / dead ball")
    touch_H, touch_A = detect_touches(ball_xy, nearest)
    control_H, control_A = filter_control_persistence(*control_candidates(nearest, speed))
    duel_zone = detect_duel_zone(nearest)
    dead_ball = detect_dead_ball(ball_xy, speed, player_xy, pitch)

    print("M2: Smart Tagging signals + restart owner")
    smart_tag_team, smart_tag_dead = load_smart_tagging_phases(smart_tagging, n_frames, team_map)
    restart_team, rstats = infer_restart_team(smart_tagging, team_map, n_frames,
                                              dead_ball, touch_H, touch_A)

    evidence = {
        'ball_xy': ball_xy,
        'ball_missing': np.isnan(ball_xy).any(axis=1),
        'touch_H': touch_H,
        'touch_A': touch_A,
        'control_H': control_H,
        'control_A': control_A,
        'duel_zone': duel_zone,
        'dead_ball': dead_ball,
        'restart_team': restart_team,
        'smart_tag_dead': smart_tag_dead,
        'smart_tag_team': smart_tag_team,
    }
    validate_evidence(evidence, n_frames)

    def pct(a):
        return "{:>6} frames ({:.1f}%)".format(int(np.sum(a)), 100.0 * np.sum(a) / n_frames)
    print("\nM2 Evidence Summary:")
    print("  ball_missing:   " + pct(evidence['ball_missing']))
    print("  touch_H:        " + pct(touch_H))
    print("  touch_A:        " + pct(touch_A))
    print("  control_H:      " + pct(control_H))
    print("  control_A:      " + pct(control_A))
    print("  duel_zone:      " + pct(duel_zone))
    print("  dead_ball:      mean {:.2f}; >=0.8 on {:.1f}%; <=0.3 on {:.1f}%".format(
        dead_ball.mean(), 100 * np.mean(dead_ball >= 0.8), 100 * np.mean(dead_ball <= 0.3)))
    print("  restart_team:   {} frames known; sources {}".format(
        int(np.sum(restart_team >= 0)), rstats))
    print("  smart_tag_team: H {} / A {} frames".format(
        int(np.sum(smart_tag_team == HOME)), int(np.sum(smart_tag_team == AWAY))))
    print("  smart_tag_dead: {} frames at 0.8".format(int(np.sum(smart_tag_dead >= 0.8))))
    return evidence


EVIDENCE_SPEC = {
    'ball_xy': (np.floating, 2), 'ball_missing': (np.bool_, 1),
    'touch_H': (np.bool_, 1), 'touch_A': (np.bool_, 1),
    'control_H': (np.bool_, 1), 'control_A': (np.bool_, 1),
    'duel_zone': (np.bool_, 1), 'dead_ball': (np.floating, 1),
    'restart_team': (np.integer, 1), 'smart_tag_dead': (np.floating, 1),
    'smart_tag_team': (np.integer, 1),
}


def validate_evidence(ev: dict, n_frames: int) -> None:
    """Assert the M2->M3 contract: keys, dtypes, lengths, value ranges."""
    assert set(ev) == set(EVIDENCE_SPEC), "keys mismatch: {}".format(set(ev) ^ set(EVIDENCE_SPEC))
    for k, (kind, ndim) in EVIDENCE_SPEC.items():
        a = ev[k]
        assert np.issubdtype(a.dtype, kind), "{} dtype {}".format(k, a.dtype)
        assert a.ndim == ndim and len(a) == n_frames, "{} shape {}".format(k, a.shape)
    for k in ('dead_ball', 'smart_tag_dead'):
        assert np.all((ev[k] >= 0) & (ev[k] <= 1)), k
    for k in ('restart_team', 'smart_tag_team'):
        assert set(np.unique(ev[k])) <= {-1, 0, 1}, k


if __name__ == "__main__":
    os.environ.setdefault("MTAH_DATA_DIR", r"C:\Hackathon\barca\data")
    evidence = main()
    print("\nM2 complete. Evidence dict ready for M3 decoder.")
