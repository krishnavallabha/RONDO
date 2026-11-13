# External data and the validation harness

Every parameter in RONDO is chosen on matches that are **not** this one. This file is the
provenance record for those matches: where each byte comes from, what its licence allows, and
how a label on a frame was decided. It is the source for the provenance table in the technical
document (Nov 16).

Nothing here is committed. Downloads land under `data/external/` (gitignored), or wherever
`RONDO_EXTERNAL_DIR` points.

```bash
python -m rondo.harness fetch      # ~2.6 GB, mostly IDSSE positional XML
python -m rondo.harness labels     # external data -> labels.csv per match
python -m rondo.harness status     # what is downloaded and built
```

## Provenance

| | IDSSE / Sportec | Metrica sample |
|---|---|---|
| **What** | 7 Bundesliga matches, 2022/23 | 2 matches, anonymised |
| **Source** | figshare [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177) | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) @ `e706dd5` |
| **Paper** | Bassek et al. (2024), *Scientific Data* | — |
| **Licence** | CC BY 4.0 | CC BY-NC 4.0 (non-commercial; this is a hackathon entry) |
| **Rate** | 25 fps | 25 fps |
| **Pitch** | 105 × 68 m, centred origin | normalised [0, 1], 105 × 68 m |
| **Possession labels** | **read** from the provider: `BallStatus`, `BallPossession` on every ball frame | **derived** by us from the hand-coded event CSV |
| **Used for** | held-out validation; the keep-or-drop decision rule | learning Metrica's conventions; a sanity anchor |
| **Pinned by** | figshare file id + md5 + byte size | commit sha + md5 + byte size |

Both rates match the tactical video's 25 fps, so an external frame and a submission frame are
the same unit of time and nothing is ever resampled.

### Why two datasets with different label quality

IDSSE labels are the provider's own per-frame channels, so they are the closest thing to an
independent ground truth we can get, and they are what the decision rule is applied to.

Metrica labels are derived from events, so they are only as good as the rules in
`rondo/harness/metrica.py`. They are not a second opinion — they are a model of *Metrica's*
conventions. That is exactly their value: Metrica wrote both this sample and this challenge's
ground truth, so a convention learned here (when a possession switches, where a dead ball
starts and ends) is a convention that applies to the match we are scored on. A parameter is
never kept on Metrica evidence alone.

## How a label is decided

**IDSSE** (`rondo/harness/idsse.py`) — nothing is inferred:

| Frame | Label |
|---|---|
| `BallStatus="0"` | `D` |
| `BallStatus="1"`, `BallPossession="1"` | `H` |
| `BallStatus="1"`, `BallPossession="2"` | `A` |

`BallPossession="1"` means the home team by DFL convention. The converter does not take that on
trust: it scores the per-frame labels against the independent event file (which names teams by
id) and refuses to write a match where the two agree on less than 60% of events — which is what
a flipped mapping looks like. The agreement is recorded in each match's `meta.json` as
`event_agreement`.

**Metrica** (`rondo/harness/metrica.py`) — derived, so the rules are stated in full:

- **Possession is last touch.** A pass in flight belongs to the team that played it, so a
  turnover lands on the opponent's `RECOVERY` / `CHALLENGE`-won frame, not on the earlier
  `BALL LOST` frame.
- **A stoppage starts when play physically stops**: the *end* frame of a `BALL OUT` (the ball
  crossing the line, not the kick), the end frame of a goal-scoring shot, the frame of a
  `FAULT RECEIVED`.
- **A stoppage ends on the restart kick**: the *start* frame of the next `SET PIECE`. That
  frame is alive and belongs to the restarting team.
- A period ending mid-stoppage stays `D` to its last tracked frame; frames with no tracking row
  are `X`.

These rules are the thing to argue with when a decoder result looks wrong on Metrica — they are
a reading of Metrica's events, not a fact about them.

## What a built match looks like

Each match becomes a folder under `data/external/labels/<dataset>/<match>/`:

| File | Contents |
|---|---|
| `labels.csv` | `frame,label` over `H/A/D/X`, 0-based, `\n` endings — the exact shape of a submission, read back through the official validator before it is written |
| `truth.npz` | `ball_xy` (metres, 105 × 68, corner origin), `ball_missing`, `half`, `frame_n` (the provider's own numbering) |
| `meta.json` | teams, pitch, label source, and the summary below |

Storing external matches in submission shape is deliberate: the harness scores them with
`possession_metrics.evaluate`, the organisers' own file, imported and never reimplemented, so a
number from the harness means the same thing as a number from the leaderboard.

All nine matches, downloaded, md5-verified and built:

| Match | Fixture | Frames | Alive | Home | Possessions | Mean possession | Event agreement |
|---|---|---|---|---|---|---|---|
| idsse J03WMX | Köln–Bayern | 145,967 | 58% | 46% | 511 | 6.7 s | 87% of 1251 |
| idsse J03WN1 | Bochum–Leverkusen | 141,561 | 48% | 53% | 500 | 5.5 s | 80% of 963 |
| idsse J03WOH | Düsseldorf–Regensburg | 137,214 | 52% | 54% | 402 | 7.2 s | 82% of 959 |
| idsse J03WOY | Düsseldorf–Rostock | 142,536 | 54% | 50% | 456 | 6.7 s | 76% of 1040 |
| idsse J03WPY | Düsseldorf–Nürnberg | 146,211 | 56% | 63% | 421 | 7.7 s | 85% of 1083 |
| idsse J03WQQ | Düsseldorf–St. Pauli | 142,345 | 59% | 44% | 401 | 8.4 s | 82% of 1116 |
| idsse J03WR9 | Düsseldorf–Kaiserslautern | 146,810 | 58% | 62% | 370 | 9.2 s | 87% of 1047 |
| metrica Sample_Game_1 | anonymised | 145,006 | 60% | 51% | 331 | 10.5 s | — (derived) |
| metrica Sample_Game_2 | anonymised | 141,156 | 68% | 53% | 349 | 11.1 s | — (derived) |

**The home/away mapping check passed on real data.** Every IDSSE match agrees with its own event
file on 76–87% of the events that land on an alive frame; a flipped `BallPossession` would score
the complement of that, around 20%. The remainder is honest disagreement between a per-frame
channel and an event clock. No frame gaps anywhere: `max_frame_gap` is 1 in all seven.

### What these matches are not

Two caveats that change how the sheet should be read, both found while building the labels:

1. **IDSSE segments possession about twice as finely as the target.** Mean possession is
   5.5–9.2 s on IDSSE, 10.5–11.1 s on the Metrica sample, and **14.2 s** in this challenge's own
   published window metadata (`BAR-ATH_possession_gt_public_meta.json`). spellF1 scores whole
   possessions, so a decoder sticky enough to produce 14-second possessions is *structurally*
   penalised on IDSSE. Calibrate stickiness and minimum durations on Metrica, whose conventions
   come from the same author as the target labels, and read IDSSE as a generalisation check —
   relative comparisons between variants, not an absolute score to maximise.
2. **The seven IDSSE matches are not seven independent samples.** Five are 2. Bundesliga matches
   with Fortuna Düsseldorf at home; the other two are Bundesliga matchday 34. Averaging spellF1
   over them understates match-to-match variance, so a decision resting on the mean should also
   hold per match before it is trusted.

### Download integrity

Worth knowing, because it happened on the first run: one positional file arrived **7.5 MB longer
than its pinned size** after two downloaders raced on the same `.part`, and another stopped at
46% when the socket timed out. Both were caught by the size and md5 gate rather than by anything
downstream — a duplicated-prefix XML would still have parsed, and would have quietly shifted
every frame index after the splice. `fetch_file` now resumes with a Range request, retries a
stalled read up to 12 times, and restarts the `.part` rather than appending if the server ignores
the range, which is what produced the long file in the first place.

## The corruption model

External matches come with a ball the provider tracked almost perfectly. This match's ball is
missing on about 41% of graded frames, in long stretches. Tuning on a clean ball therefore
answers a question nobody asked, so every external ball is corrupted before the pipeline sees it.

`rondo/harness/noise.py` splits that into a measurement and a replay:

```bash
python -m rondo.harness noise --match Sample_Game_1 --out profiles/metrica_native.json
python -m rondo.harness noise --evidence out/evidence.npz --out profiles/atd_BAR-ATH.json
python -m rondo.harness run --profile metrica_native --clean --bootstrap 400
```

A **profile** is a few hundred bytes of JSON: missing rate, the gap-length and visible-length
distributions as 21 percentiles each, and position jitter in metres per axis. It holds no data
and no frame positions — only how long the ball goes missing, how often, and how far apart. That
is what makes it committable under `profiles/`, and it is the reason this path exists next to
`rondo/signals/m2_validate.py`: that one transplants this match's actual missing *mask*, so it
only runs where the gated bundle is unpacked, while a profile lets a clean clone (or CI)
reproduce every external number with no match data at all.

The committed stand-in, `profiles/metrica_native.json`, is measured over both Metrica sample
matches — the same provider as this match, at the same 25 fps:

| | measured |
|---|---|
| missing rate | **40.1%** (this match: ~41% of graded frames) |
| gaps | 323 over 286,162 frames |
| gap length | median 155 frames (6.2 s), mean 355, max 3,292 (132 s) |
| missing frames in gaps ≥ 10 s | **92%** (measured on this match by Signals: 85%) |
| jitter | 0.001 m — Metrica ships this sample already smoothed |

Replaying it draws gap lengths from the measured distribution until they account for exactly the
measured rate, then splits the remaining frames into visible runs drawn from the measured
distribution and scaled to fit. Two deliberate choices:

- **Gaps are stretches, never a per-frame coin flip.** A Bernoulli mask at 41% is thousands of
  one-frame holes, which M1 bridges trivially; it is not this match's failure mode.
- **The rate is exact on every seed.** Drawing both run types freely left the achieved rate
  anywhere between 34% and 43% on a single match, which would surface as a difference between
  decoder variants that were scored on different seeds. The cost is that the *spacing* between
  gaps is the measured shape stretched to fit, which the decoder does not feel the way it feels
  gap length and total rate.

Two honest limitations:

- **Placement is blind to play state.** On this match a missing ball mostly *is* a stoppage; the
  profile reproduces the marginal gap statistics, not that correlation, so a long gap can land
  in the middle of live play. That makes this condition harder than reality, not easier — fine
  for a keep-or-drop rule, wrong for estimating an absolute score. The correlated version is
  `deadmiss` in `rondo/signals/m2_validate.py`.
- **No jitter until the ATD profile is measured.** The stand-in carries none because Metrica's
  sample is pre-smoothed. The one command above fills that in on a machine with the data.

## The results sheet

`python -m rondo.harness score` appends one row per (experiment, match) to
`data/external/results.csv`: spellF1 mean and at each IoU, the block-bootstrap interval,
macro-F1 and per-class F1 (including the D class), predicted vs. true possession counts, and
median/p90 boundary error in frames.

**Decision rule.** A module or parameter is kept only when it improves mean held-out spellF1 by
more than the bootstrap noise band — `possession_metrics.SPELL_NOISE_BAND`, 0.04. That is
`score.keep(before, after)`, and it is not a judgement call.

### Where the pipeline stands, Nov 13

`python -m rondo.harness run --profile metrica_native --clean --bootstrap 400`, 400 bootstrap
draws, seed 0, nine matches, no match data involved anywhere:

| Experiment | IDSSE (7) | worst match | Metrica (2) | macroF1 | D-class F1 | spells predicted / true |
|---|---|---|---|---|---|---|
| `harness_metrica_native` (realistic ball) | **0.363** | 0.325 | 0.383 | 0.728 | 0.834 | 1,703 / 3,061 |
| `harness_clean` (provider ball) | 0.509 | 0.479 | 0.521 | 0.883 | 0.976 | 2,563 / 3,061 |

**The Nov 13 milestone is met with room to spare.** Every single match clears the 0.2433 floor
under the realistic ball condition, worst case 0.325 on Köln–Bayern.

Two independent confirmations, which is the point of having both external paths:

- Signals' own run (`m2_validate`, which transplants this match's actual missing mask and needs
  the gated data) reports held-out IDSSE 0.37 corrupted / 0.51 full. This path, with a measured
  profile and no match data, gets 0.363 / 0.509. Two different corruption models agreeing to
  within 0.01 is worth more than either number alone.
- The lag baselines put that score in context: 0.363 sits between `truth_lag_50` (0.356) and
  `truth_lag_25` (0.557), so the pipeline's total error currently costs about what a **two-second
  systematic boundary error** would. That is the argument for spending Nov 14 on boundary
  snapping rather than on emissions.

**The failure mode is under-segmentation.** Under the realistic ball the decoder commits to 1,703
possessions where the truth has 3,061 — it merges. Some of that is the IDSSE granularity caveat
above (their possessions are about twice as fine as this challenge's), but Metrica, whose
conventions are the target's, shows the same direction: 416 predicted against 680. A missing ball
removes the touch evidence that would have split two possessions, and minimum durations then hold
the merge in place.

### Reference predictors

Not models — they exist so the sheet records what a floor and a near-ceiling look like, and
because a first real number is unreadable without them.

| Experiment | IDSSE (7) | Metrica (2) | All |
|---|---|---|---|
| `constant_home` (everything H) | 0.000 | 0.000 | 0.000 |
| `truth` | 1.000 | 1.000 | 1.000 |
| `truth_lag_12` (every boundary half a second late) | 0.775 | 0.790 | 0.778 |
| `truth_lag_25` (one second late) | 0.557 | 0.621 | 0.571 |
| `truth_lag_50` (two seconds late) | 0.356 | 0.454 | 0.377 |

**Read the lag rows before building anything else.** Perfect labels with every boundary half a
second late lose ~0.22 spellF1; a full second late loses ~0.43. Timing is not a polish step on
this metric — it is most of the score. The penalty is steeper on IDSSE than on Metrica for the
reason given above: shorter possessions lose IoU faster for the same absolute lag.

`constant_home` scoring exactly 0.000 on all nine is the metric working as documented: one
match-long spell overlaps no true possession at IoU 0.3, so a degenerate submission earns
nothing, while its macro-F1 still comes out at 0.15–0.17.

### Reading the intervals

Per-match bootstrap intervals are 0.05–0.09 wide (±0.025 to ±0.045). The decision rule compares
the *mean* over seven held-out matches, whose standard error is roughly a third of that, so the
0.04 band is conservative rather than generous — a change that clears it is real.

One rule when comparing variants: **use the same corruption seed.** The missing rate is exact on
every seed, but which frames go missing is not, and a possession that happens to lose its only
touch is a coin flip worth more than 0.04 on a single match. The sheet records the seed and the
achieved missing rate in its `notes` column for exactly this reason.
