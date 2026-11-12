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

from rondo.signals.m1_ingest import main as ingest_atd, clean_ball  # noqa: E402

# Params
FPS = 25
CONTROL_RADIUS = 1.5            # m
MIN_PLAYER_DIST = 1.5           # m, touch proximity
VEL_WINDOW = 6                  # frames each side; M1's 1.25 Hz low-pass spreads a kick over ~0.5 s
TOUCH_IMPULSE = 4.0             # m/s, |v_after - v_before| (kick, trap, block)
TOUCH_DIRECTION_THRESHOLD = 30  # degrees
TOUCH_DIRECTION_MIN_SPEED = 3.0 # m/s
TOUCH_NMS = 3                   # frames, keep one touch per local impulse peak
VEL_WINDOW_SHORT = 3            # frames, fallback where the ball is not seen VEL_WINDOW away
TOUCH_TIE_MARGIN = 0.5          # m, nearest H and A within this of each other: credit both
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
MISSING_STALE_FRAMES = 750      # 30 s: restarts come well within this, so after it the last
                                # sighting of the ball says nothing about whether it is dead
TAKER_RADIUS = 3.0              # m, a throw-in taker holds the ball, a kick taker stands over it
TAKER_SEARCH = 50               # frames after the stoppage in which the ball must reappear
KICKOFF_SPOT_RADIUS = 3.0       # m around the centre spot
SHAPE_MIN_PLAYERS = 8           # per team, below this a team's spread is not measured
SHAPE_SMOOTH = 25               # frames, centred window for the spread ratio (1 s)
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
    Velocities use VEL_WINDOW frames each side, or VEL_WINDOW_SHORT where the ball is not seen
    that far away (next to a gap). Events are thinned to local impulse peaks (+-TOUCH_NMS), then
    credited to the team whose nearest player is within min_player_dist and closer than the
    opponent's (tolerating +-1 frame), or to both when the two are within TOUCH_TIE_MARGIN.
    NaN ball -> no touch.
    """
    n = len(ball_xy)
    vb, va = ball_velocity(ball_xy)
    vb_s, va_s = ball_velocity(ball_xy, VEL_WINDOW_SHORT)
    vb = np.where(np.isnan(vb), vb_s, vb)
    va = np.where(np.isnan(va), va_s, va)
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
    dH, dA = nd_tol[:, HOME], nd_tol[:, AWAY]
    with np.errstate(invalid="ignore"):
        tie = np.abs(dH - dA) <= TOUCH_TIE_MARGIN
    touch_H = event & (dH <= min_player_dist) & ((dH < dA) | tie)
    touch_A = event & (dA <= min_player_dist) & ((dA < dH) | tie)
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


def team_spread_ratio(player_xy: np.ndarray, player_team: np.ndarray,
                      min_players: int = SHAPE_MIN_PLAYERS,
                      smooth: int = SHAPE_SMOOTH) -> np.ndarray:
    """float[N] log(spread_H / spread_A), NaN where unknown. Works without the ball.

    Spread is the root of the summed x and y variance of a team's tracked players. The team in
    possession spreads out to make space and the defending team compacts, so positive values
    lean H and negative lean A. Frames where either team has fewer than `min_players` tracked
    players are NaN; the rest are averaged over a centred `smooth`-frame window (NaN-aware).
    This is a raw feature, not a probability: its weight is to be fitted on external matches.
    """
    n = len(player_xy)
    spreads = []
    for team in (HOME, AWAY):
        xy = player_xy[:, player_team == team, :]
        seen = ~np.isnan(xy[..., 0])
        cnt = seen.sum(axis=1)
        x = np.where(seen, xy[..., 0], 0.0)
        y = np.where(seen, xy[..., 1], 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = ((x ** 2).sum(1) / cnt - (x.sum(1) / cnt) ** 2
                   + (y ** 2).sum(1) / cnt - (y.sum(1) / cnt) ** 2)
            sp = np.sqrt(np.maximum(var, 0.0))
        sp[cnt < min_players] = np.nan
        spreads.append(sp)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.log(spreads[HOME] / spreads[AWAY])
    raw[~np.isfinite(raw)] = np.nan
    if smooth <= 1:
        return raw
    ok = ~np.isnan(raw)
    csum = np.concatenate(([0.0], np.cumsum(np.where(ok, raw, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    half = smooth // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    total, count = csum[hi] - csum[lo], ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[~ok] = np.nan
    return out


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
         (0.9 and 0.7 expire after MISSING_STALE_FRAMES of missing ball and fall back to 0.5)
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
    gl = gap_len[miss]
    fresh = gl <= MISSING_STALE_FRAMES
    p_miss = np.full(miss.sum(), 0.5)
    p_miss[near_line[lv] & (gl > MISSING_BALL_AT_BOUNDARY_GAP) & fresh] = 0.7
    p_miss[oob[lv] & fresh] = 0.9
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


def localise_restart(s: int, e: int, dead_ball, touch_H, touch_A, team=None,
                     nearest=None, ball_xy=None, centre=None):
    """(frame, team) of the restart inside a tag window [s,e), or (None, -1).

    In order:
    1. The first touch after the last dead stretch (dead_ball >= 0.7) in the window, by `team`
       when it is known.
    2. Team known and the window has a dead stretch: the frame the ball leaves it.
    3. Team unknown: whoever has a player within TAKER_RADIUS of the ball on its first visible
       frame after the stoppage (the taker holds or stands over the ball).
    4. Team known and `centre` given (kickoff): the frame the ball leaves the centre spot,
       otherwise the middle of the window (short kickoff tags bracket the restart).
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
    if team is None and nearest is not None:
        for f in range(after, min(e, after + TAKER_SEARCH)):
            d = nearest[f]
            if np.isnan(d).all():
                continue
            d = np.where(np.isnan(d), np.inf, d)
            if d.min() <= TAKER_RADIUS and d[HOME] != d[AWAY]:
                return f, (HOME if d[HOME] < d[AWAY] else AWAY)
            break
    if team is not None and centre is not None:
        if ball_xy is not None:
            near = np.linalg.norm(ball_xy[s:e] - np.asarray(centre), axis=1) <= KICKOFF_SPOT_RADIUS
            runs = _runs(near)
            if runs:
                return s + int(runs[0][1]) - 1, team
        return (s + e) // 2, team
    return None, -1


def infer_restart_team(smart_tagging: list, team_map: Dict[str, int], n_frames: int,
                       dead_ball, touch_H, touch_A, nearest=None, ball_xy=None,
                       pitch=None, kickoffs=None) -> Tuple[np.ndarray, Dict[str, int]]:
    """restart_team int8[N]: who takes each restart, on +-RESTART_PAD frames around it.

    - Kick Off (Start): `kickoffs` {half: (frame, team)}, the published frames and owners.
    - Corner / goal kick / free kick: Team tag = team performing the set piece.
    - Kick Off (After Goal): team that conceded = opposite of the preceding GOALS tag.
    - Throw-ins (Team 'N/A'): team of the first touch after the dead stretch, else the team
      standing at the ball when it reappears, from tracking.
    The restart frame inside each (mostly 35 s) tag window is localised from tracking.
    """
    restart = np.full(n_frames, -1, dtype=np.int8)
    stats = {"kickoff": 0, "tag": 0, "after_goal": 0, "tracking": 0, "unlocated": 0}
    goals = sorted((p['start_frame'], team_map.get((p['tags'].get('Team') or '').strip()))
                   for p in smart_tagging if (p['code'] or '').strip().lower() == 'goals')

    def mark(f, team):
        restart[max(0, f - RESTART_PAD):min(n_frames, f + RESTART_PAD + 1)] = team

    for f, team in (kickoffs or {}).values():
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
        centre = (pitch[0] / 2, pitch[1] / 2) if pitch is not None and 'kick off' in typ else None
        f, found = localise_restart(s, e, dead_ball, touch_H, touch_A, team,
                                    nearest=nearest, ball_xy=ball_xy, centre=centre)
        if f is None:
            stats["unlocated"] += 1
            continue
        stats[src if team is not None else "tracking"] += 1
        mark(f, found)
    return restart, stats


# ---------------------------------------------------------------- pipeline

def build_evidence(ball_xy: np.ndarray, player_xy: np.ndarray, player_team: np.ndarray,
                   pitch: Tuple[float, float], fps: int = FPS, clean: bool = True,
                   smart_tagging: list = None,
                   kickoffs: dict = None) -> Tuple[Dict[str, np.ndarray], dict]:
    """M2 evidence for ANY match from plain arrays. Returns (evidence, info).

    The entry point for external matches (the harness): one match's tracking in, the same keys
    this match produces out.

    ball_xy      float [N, 2] metres, NaN where the ball is not tracked.
    player_xy    float [N, P, 2] metres, NaN where a slot is empty. Any number of slots P.
    player_team  int [P], 0 = home (H), 1 = away (A).
    pitch        (length, width) in metres. x runs 0..length and y 0..width from a pitch corner
                 (for centre-origin data add length/2 and width/2 first).
    fps          must be 25: every threshold here is per frame at 25 fps. Resample first.
    clean        apply the M1 ball cleaning (pass False if ball_xy is already cleaned).
    smart_tagging, kickoffs   this match only. Without them smart_tag_* and restart_team are
                 unknown everywhere (0.5 / -1), which the contract allows.
    """
    ball_xy = np.asarray(ball_xy, dtype=float)
    player_xy = np.asarray(player_xy, dtype=float)
    player_team = np.asarray(player_team)
    n = len(ball_xy)
    if fps != FPS:
        raise ValueError("fps must be {} (got {}): resample the tracking first".format(FPS, fps))
    if (ball_xy.shape != (n, 2) or player_xy.ndim != 3 or player_xy.shape[0] != n
            or player_xy.shape[2] != 2 or player_xy.shape[1] != len(player_team)):
        raise ValueError("expected ball_xy [N,2], player_xy [N,P,2], player_team [P]; got {}, {}, {}"
                         .format(ball_xy.shape, player_xy.shape, player_team.shape))
    if not set(np.unique(player_team).tolist()) <= {HOME, AWAY}:
        raise ValueError("player_team must contain only 0 (home) and 1 (away)")
    length, width = pitch
    px, py = player_xy[..., 0], player_xy[..., 1]
    seen = ~np.isnan(px)
    if seen.any():
        outside = ((px[seen] < -10) | (px[seen] > length + 10)
                   | (py[seen] < -10) | (py[seen] > width + 10))
        if outside.mean() > 0.05:
            raise ValueError("{:.0%} of player positions lie outside the pitch: coordinates must be "
                             "metres from a pitch corner, 0..{} by 0..{}"
                             .format(outside.mean(), length, width))

    info = {}
    if clean:
        ball_xy, info["clean_stats"] = clean_ball(ball_xy, pitch)

    speed = ball_speed_mps(ball_xy)
    nearest = team_nearest(ball_xy, player_xy, player_team)
    touch_H, touch_A = detect_touches(ball_xy, nearest)
    control_H, control_A = filter_control_persistence(*control_candidates(nearest, speed))
    dead_ball = detect_dead_ball(ball_xy, speed, player_xy, pitch)

    if smart_tagging:
        team_map = infer_team_mapping(smart_tagging)
        if set(team_map.values()) != {HOME, AWAY}:
            print("  WARNING: Team tags not mapped for both sides: {}".format(team_map))
        smart_tag_team, smart_tag_dead = load_smart_tagging_phases(smart_tagging, n, team_map)
    else:
        team_map = {}
        smart_tag_team = np.full(n, -1, dtype=np.int8)
        smart_tag_dead = np.full(n, 0.5)
    restart_team, info["restart_stats"] = infer_restart_team(
        smart_tagging or [], team_map, n, dead_ball, touch_H, touch_A,
        nearest=nearest, ball_xy=ball_xy, pitch=pitch, kickoffs=kickoffs)

    evidence = {
        'ball_xy': ball_xy,
        'ball_missing': np.isnan(ball_xy).any(axis=1),
        'touch_H': touch_H,
        'touch_A': touch_A,
        'control_H': control_H,
        'control_A': control_A,
        'duel_zone': detect_duel_zone(nearest),
        'dead_ball': dead_ball,
        'restart_team': restart_team,
        'smart_tag_dead': smart_tag_dead,
        'smart_tag_team': smart_tag_team,
        'shape_spread': team_spread_ratio(player_xy, player_team),
    }
    validate_evidence(evidence, n)
    return evidence, info


def summarise(evidence: Dict[str, np.ndarray], info: dict) -> None:
    """Print aggregate counts only, never rows."""
    n = len(evidence['ball_missing'])

    def pct(a):
        return "{:>6} frames ({:.1f}%)".format(int(np.sum(a)), 100.0 * np.sum(a) / n)
    db, sp = evidence['dead_ball'], evidence['shape_spread']
    print("\nM2 Evidence Summary:")
    for key in ('ball_missing', 'touch_H', 'touch_A', 'control_H', 'control_A', 'duel_zone'):
        print("  {:<15} {}".format(key + ":", pct(evidence[key])))
    print("  dead_ball:      mean {:.2f}; >=0.8 on {:.1f}%; <=0.3 on {:.1f}%".format(
        db.mean(), 100 * np.mean(db >= 0.8), 100 * np.mean(db <= 0.3)))
    print("  restart_team:   {} frames known; sources {}".format(
        int(np.sum(evidence['restart_team'] >= 0)), info.get("restart_stats")))
    print("  smart_tag_team: H {} / A {} frames".format(
        int(np.sum(evidence['smart_tag_team'] == HOME)),
        int(np.sum(evidence['smart_tag_team'] == AWAY))))
    print("  smart_tag_dead: {} frames at 0.8".format(int(np.sum(evidence['smart_tag_dead'] >= 0.8))))
    print("  shape_spread:   known on {:.1f}% of frames, {:.1f}% of ball-missing frames".format(
        100 * np.mean(~np.isnan(sp)), 100 * np.mean(~np.isnan(sp[evidence['ball_missing']]))))


def main(game: str = GAME, data_dir: str = None, m1_data: dict = None) -> Dict[str, np.ndarray]:
    """Evidence for THIS match: tactical ATD, its Smart Tagging and the published kickoffs."""
    if m1_data is None:
        print("M2: Loading M1 output")
        m1_data = ingest_atd(game, data_dir)
    print("M2: Loading Smart Tagging")
    smart_tagging = load_smart_tagging(game, data_dir)
    print("M2: Building evidence")
    evidence, info = build_evidence(m1_data['ball_xy'], m1_data['player_xy'],
                                    m1_data['player_team'], GAMES[game]["pitch"], clean=False,
                                    smart_tagging=smart_tagging, kickoffs=KICKOFFS)
    if len(evidence['ball_missing']) != source_frames("tactical", game):
        raise ValueError("evidence length does not match the tactical video")
    summarise(evidence, info)
    return evidence


EVIDENCE_SPEC = {
    'ball_xy': (np.floating, 2), 'ball_missing': (np.bool_, 1),
    'touch_H': (np.bool_, 1), 'touch_A': (np.bool_, 1),
    'control_H': (np.bool_, 1), 'control_A': (np.bool_, 1),
    'duel_zone': (np.bool_, 1), 'dead_ball': (np.floating, 1),
    'restart_team': (np.integer, 1), 'smart_tag_dead': (np.floating, 1),
    'smart_tag_team': (np.integer, 1),
    'shape_spread': (np.floating, 1),
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
