# Signals: evidence and learned emissions

Section of the RONDO technical document. Owner: Signals (A). Code: `rondo/signals/`. Full detail in
`docs/signals.md`; decoder in `docs/decoder.md`.

## 1. What this part does

Signals turns the provided tactical tracking (ATD) and Smart Tagging into one evidence file with a
row per tactical video frame (199,319 rows). Its format is fixed by `rondo/contract.py` and checked
on every run. The decoder reads only this file. The submitted run uses two learned per-frame
probabilities: `p_dead` (the ball is out of play) and `p_home` (Barcelona has the ball, given it is
alive). The decoder (explicit-duration HMM, chain Viterbi) turns them into H/A/D/X labels.

```
python -m rondo.signals.export --out out/evidence.npz
python -m rondo.run --evidence out/evidence.npz --out out/RONDO.csv
```

## 2. Pipeline

**M1, ingest and ball cleaning.** The ATD is on the tactical video's timeline
(`video_frame = frame_id - 1`), scaled to the real pitch (105.3 × 68.0 m, `games.py`). The ball
track is cleaned by fixed physical rules:

- drop positions more than 10 m off the pitch;
- cut at jumps faster than 40 m/s;
- drop short excursions (at most 2 s) after which the track reconnects;
- interpolate gaps up to 0.5 s, or up to 2 s when entry and exit velocities agree;
- smooth each continuous stretch with a zero-phase 1.25 Hz filter.

The raw ATD has a ball on 56.9% of frames. After cleaning, 55.6% are usable.

**M2, evidence.**
- **Touches:** velocity or direction changes of the ball, credited to a team by proximity (1.5 m).
- **Control:** a team's nearest player within 1.5 m of a slow ball for at least 5 frames.
- **Dead-ball score:** ball out of the pitch, ball still with slow players, ball missing after
  last being seen out or near a line, ball missing with slow players.
- **Smart Tagging:** team tags and set-piece tags, used as soft priors only.
- **Restart owner:** for 53 of the 57 restarts; kickoff owner from Law 8 positions.

One finding about our match drives the dead-ball evidence. The ball is missing on 41% of graded
frames, 85% of those frames sit in gaps longer than 10 s, and players move at 1.00 m/s while it is
missing against 1.94 m/s while it is seen. So a missing ball mostly means a stoppage, the same
pattern Metrica's own tracking shows on the external matches.

**M7, learned emissions** (`m7_learned.py`). Two logistic regressions in plain numpy:

- **Dead model** (54 features): whether the ball is missing and for how long, position inside the
  gap, player speed divided by this match's median, the dead-ball levels, ball still / out / fast,
  kicks, distance to the lines.
- **Team model** (50 features, no intercept):
  - soft ball-to-team proximity over windows from 1 frame to 12 s;
  - sustained control;
  - credited touches and any-kick touches, as past and future decays;
  - team width.

  Every team-model feature is home minus away, and every row is trained from both viewpoints, so
  swapping the teams gives exactly `1 − p_home`. Rows are weighted so each possession counts about
  equally (a run of n frames weighs clip(200/n, 0.25, 4)), because spellF1 counts possessions, not
  frames.
- **Weights** are committed (`m7_weights.json`, 8 kB). `python -m rondo.signals.m7_learned train`
  refits them from the external data and reproduces the same file.

## 3. Training and validation data

All labels come from other matches:

| Data | Matches | Labels |
|---|---|---|
| IDSSE Bundesliga open data | 7 | provider's per-frame ball status and possession |
| Metrica Sports sample data (Games 1–2) | 2 | derived from Metrica's hand-coded events with fixed rules |

Provenance and licences are in `docs/EXTERNAL_DATA.md`.

External balls are much cleaner than ours, so every match is used under five conditions:

| Condition | What changes |
|---|---|
| `full` | the provider's ball, unchanged |
| `deadmiss` | 78% of stoppages and 13% of contacts hidden, the way Metrica's tracking loses the ball |
| `n10dm` | `deadmiss` plus 1.0 m of slow player-position noise (4 s time scale) |
| `n16dm` | `deadmiss` plus 1.6 m of the same noise |
| `n13corrupt` | our match's missing-ball mask transplanted, plus 1.3 m noise |

Training uses every 4th graded frame.

**Protocol.**
- Models never see the match they are scored on: Metrica is held out as a block, IDSSE in three
  folds.
- Scoring uses the organisers' `possession_metrics`, imported and not reimplemented, through our
  real decoder.
- A change is kept only if mean held-out spellF1 rises by more than 0.04 (the scorer's noise band),
  checked per condition.

## 4. Results

Mean held-out spellF1 over the 5 conditions:

| Emissions | Metrica | IDSSE |
|---|---|---|
| Heuristic (M2 evidence only) | 0.406 | 0.378 |
| **M7 learned (submitted)** | **0.485** | **0.485** |

By condition (M7, in the order full / deadmiss / n10dm / n16dm / n13corrupt):
- Metrica: 0.614 / 0.521 / 0.474 / 0.433 / 0.386
- IDSSE: 0.596 / 0.528 / 0.489 / 0.437 / 0.377

**Our match, label-free:** 37.8% dead, 35.4% home, 26.9% away (graded frames); 406 possessions.

**Public 5 minutes.** We scored them only as a correctness check (format, alignment, scorer wiring)
and never used them to choose anything. With 17 possessions the scorer's own noise band on the
window is about ±0.28.

## 5. Tried and not used

Every row was tested on the harness above and fell short of the 0.04 keep rule, or could not be
validated on the external matches.

| Idea | Result |
|---|---|
| Ball-free team-shape features (spread, back-line height, marking gap, read ahead) | IDSSE −0.003 to −0.07; best +0.005 |
| Wider touch radius / time window | +0.004 to +0.014 |
| Wider control radius (2–3 m) | 0 to −0.07 |
| Player-speed dead ball everywhere | −0.03 |
| Missing ball + fast players = alive | −0.06 to −0.09 |
| Strict control (no opponent within 1.5 m, dwell) | −0.02 to −0.05 |
| Last-owner carry; best combination of the above | +0.02 / +0.03 |
| M7 plus collective player speed and ball-moves-with-player features | +0.001 / +0.007 |
| Explicit model of the dead share inside ball gaps | −0.03 to +0.01 |
| 20 SkillCorner open matches (A-League, MIT) added to training | Metrica ±0.003, IDSSE −0.01 |
| Milder possession weights (1/√n) / no weights | +0.014 Metrica, −0.006 IDSSE / −0.02 to −0.03 |
| Training labels under the challenge's definition (lone touches and spells under 1 s are not changes) | within ±0.006 |
| Metrica Sample Game 3 added to training | +0.003 or less |
| Broadcast-tracking ball fill (broadcast synced to tactical at +4,014 / +26,921 frames, under 1 frame error) | fills 6.2% of missing frames; the two balls disagree by more than 3 m on 17.6% of shared frames |
| Broadcast "is the feed showing the game" dead-ball cue | real signal (73% of our dead frames off-air, 18% of alive), changes 5% of frames, no external match to score it on |
| Ball detection on the video (two public YOLO models) | dead share falls to 22%, outside every external match (32–55%); cannot be scored externally |

No code in the repository reads the video.

## 6. Rules and disclosure

- **No label of this match is used.** No hand labels and no human-produced data about the match
  from any source. The full-match ground truth is never read. The public 5 minutes are a
  correctness check only.
- **Everything learned or tuned is fitted on the external matches.** Two label-free quantities
  come from this match's tracking:
  - its missing-ball mask, transplanted into one training condition;
  - its median player speed, used as a unit.

  The slow-players rule was motivated by one aggregate of the tracking (players' speed with the
  ball missing against seen), and its value was checked on the external matches.
- **Smart Tagging** is used as automatic input, as the FAQ allows. The only frame constants are
  the kickoff frames published by the organisers.
- **AI assistant.** Only aggregate numbers printed by our own scripts were shared with it. No
  match data, extract or frame-level file was uploaded to any AI service.
- **No match data or frame-level derived data is in the repository.** `.gitignore` excludes
  `data/`, `out/` and `*.npz`. The committed weights are 105 coefficients plus feature means and
  scales, fitted on external data.

## 7. Limitations

- No external condition reproduces our ball exactly. For example, touches with a player within
  1.5 m are three times rarer on our match than on any external match.
- The restart pull the decoder applies (about 1,050 frames) uses Smart Tagging, which the external
  matches do not have, so it is untested.
- IDSSE splits possession about twice as finely as the target definition, and 5 of its 7 matches
  share a home team.
