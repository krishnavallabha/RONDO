# Signals (M0–M2): from tracking to per-frame evidence

Draft section of the RONDO technical document. Owner: Signals (A). Code: `rondo/signals/`.

## Role

Signals turns the tactical tracking (ATD) and Smart Tagging into one evidence file,
`evidence.npz`, with one row per tactical video frame (199,319). The decoder (M3) reads only this
file. Its format is fixed by `rondo/contract.py` and checked on every run.

```
python -m rondo.signals.export --out out/evidence.npz     # M1 + M2, under a minute
python -m rondo.run --evidence out/evidence.npz --out out/RONDO.csv
```

Every value comes from code. Nothing is labelled, corrected or tuned by hand, and nothing is fitted
to this match's ground truth.

## M0: hidden channels

The EPTS export could, in principle, carry Metrica's own possession signal. We checked:
`ball_state` exists but is empty, and `ball_owning_team` is absent. Possession therefore has to be
inferred from positions alone.

## M1: ingest and ball cleaning

**Alignment and units.** The ATD is already on the tactical video's timeline, so
`video_frame = frame_id - 1` (`starter/load_atd.py`). Coordinates are normalised and scaled to the
real pitch, 105.3 × 68.0 m (`games.py`), not the 105 × 68 in the ATD metadata. Slot-to-team mapping
comes from the ATD XML (first team = home = Barcelona).

**Ball cleaning**, in order:

| Step | Rule | Why |
|---|---|---|
| Out of pitch | drop positions more than 10 m beyond a line | a real ball out of play is a few metres out; tens of metres is tracking junk |
| Impossible jumps | cut the track wherever it moves faster than 40 m/s (1.6 m per elapsed frame) | the hardest shots are about 35 m/s |
| Excursions | drop a cut segment when the track on both sides reconnects without it and it is no longer than either neighbour | the tracker briefly followed something else and came back |
| Relocations | keep segments that do not reconnect, with one blank frame at the jump | the tracker re-locked on the ball; nothing is interpolated or smoothed across the jump |
| Short gaps | interpolate gaps of up to 12 frames (0.5 s) | brief occlusions |
| Longer gaps | interpolate up to 50 frames (2 s) only if entry and exit velocity agree (within 30° and a factor of 3 in speed) | a pass in flight; otherwise the ball may have been played |
| Smoothing | zero-phase 2nd-order Butterworth low-pass at 1.25 Hz, run separately on each continuous stretch of at least 16 frames | removes jitter without the time lag of a one-way filter, and never across a gap |

On this match: the ball is in the raw ATD on 56.9% of frames; 661 positions are dropped as out of
pitch, 2,603 as impossible jumps, 61 gap frames are interpolated, leaving 55.3% usable.

An earlier version compared each frame with the last accepted position. It discarded about 2,860
good frames whenever the tracker re-locked on the ball (up to 24 frames each time) and interpolated
across relocations. The segment rule replaced it.

## M2: evidence

All thresholds are per frame at 25 fps.

**Touches** (`touch_H`, `touch_A`). A ball event is a velocity change of more than 4 m/s between
the 6 frames before and after (3 frames next to a gap), or a turn of more than 30° while the ball
moves faster than 3 m/s. Events are thinned to one per local peak (±3 frames). The touch goes to
the team whose nearest player is within 1.5 m and closer than the opponent's nearest (tolerating
±1 frame), or to both when the two are within 0.5 m of each other. Touches are sparse (about 400
events) and are best used as boundary hints.

**Control** (`control_H`, `control_A`). A team's nearest player is within 1.5 m of the ball and
closer than the opponent's, and the ball is slower than 10 m/s (carried, not travelling). The run
must last at least 5 frames, allowing dropouts of up to 2 frames. This filters evidence; minimum
possession length is the decoder's job.

**Duel zone.** Both teams have a player within 1.5 m of the ball.

**Dead ball** (`dead_ball`), a heuristic score, not a calibrated probability:

| Value | Condition |
|---|---|
| 1.0 | ball more than 0.5 m outside the pitch |
| 0.9 | ball missing, last seen out of the pitch (for at most 30 s) |
| 0.8 | ball still (under 0.3 m/s for 0.5 s) and players slow (mean under 1.5 m/s) |
| 0.7 | ball missing over 2 s after last being seen within 2 m of a line (for at most 30 s) |
| 0.3 | ball moving faster than 3 m/s inside the pitch |
| 0.5 | otherwise (unknown) |

The 30 s expiry matters: without it, one lost ball held "dead" for 2.3 minutes. Restarts come well
within 30 s, so after that the last sighting says nothing.

**Smart Tagging priors.** `smart_tag_team` copies each phase's Team tag (Barcelona = H; the file
spells Athletic "Atletic Club"). `smart_tag_dead` is 0.8 on short set-piece tags (kickoffs, about
5 s) and 0.6 on the long ones (most set-piece tags are a fixed 35 s window, much of it live play).
The tags are inferred automatically, so they are soft priors only.

**Restart owner** (`restart_team`), marked on ±10 frames around each restart:

- Kickoffs at the start of each half: frames as published in the challenge docs. The owner is read
  from tracking (Law 8: only the kicking team may be at the centre spot, opponents stay 9.15 m away).
  If only one half is readable, the other is the opposite team (Law 8 alternation).
- Corners, goal kicks, free kicks: the phase's Team tag; the frame is found from tracking (first
  touch after the last dead stretch in the tag window).
- Kickoffs after a goal: the team that conceded; frame where the ball leaves the centre spot.
- Throw-ins (no team in the tag): the team of the first touch, else the team with a player within
  3 m of the ball when it reappears.

53 of 57 restarts are located.

**Ball-free feature** (`shape_spread`, proposed). The log ratio of how spread out the two teams
are (root of the summed x and y variance), smoothed over 1 s. The team in possession spreads out
and the defender compacts. It uses only player positions, so it exists when the ball is missing,
which is about 44% of frames. On this match it separates H from A control with AUC 0.72, measured
against our own control evidence, not the ground truth. It is a raw feature; its weight is to be
fitted on the external matches. Three other shape features (team tilt, centroid gap, forward
speed gap) showed no signal (AUC 0.47–0.54) and were not added.

## External matches

`build_evidence(ball_xy, player_xy, player_team, pitch)` produces the same evidence for any match
from plain arrays (metres from a pitch corner, 25 fps), so the harness runs the exact same signals
on labelled external matches. It rejects other frame rates and centre-origin coordinates.

## Tried and not used

**Broadcast ball fill (M4).** The broadcast recording has its own timeline with cuts and replays.
Matching player positions finds it exactly: tactical frame = broadcast + 4,014 in the first half
and + 26,921 in the second, under one frame of error, with replays as short detours. But the
broadcast ball would fill only 6.2% of graded frames where the tactical ball is missing (it is
usually lost at the same moments), the two balls disagree by more than 3 m on 17.6% of shared
frames, and the external matches have no second recording to validate it on. Not used.

**Video.** No code reads either video file.

## Reproducibility and checks

- Pinned environment (Python 3.13, `requirements.txt` including pandas' own dependencies). Two
  runs give the same CSV hash, also identical to a Python 3.15 run.
- 41 unit tests on synthetic data (no match data needed) run in CI: ball cleaning, touches, control,
  dead ball, restarts, kickoff owner, the shape feature, the external-match entry point and the
  contract.
- The public 5-minute ground truth was used only to score the output as a correctness check. No
  setting was chosen by its score.

## Known limitations

- Touch attribution accuracy is unmeasured; it needs the labelled external matches.
- `dead_ball` and `shape_spread` are uncalibrated scores until fitted on the external matches.
- 4 restarts are not located (no touch, no dead stretch, nobody at the ball).
