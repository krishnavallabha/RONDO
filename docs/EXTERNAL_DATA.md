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

Built on Nov 12:

| Match | Frames | Alive | Home share | Possessions | Mean possession | Native ball-missing |
|---|---|---|---|---|---|---|
| Metrica Sample_Game_1 | 145,006 | 60% | 50.6% | 331 | 10.5 s | **39.1%** |
| Metrica Sample_Game_2 | 141,156 | 68% | 52.6% | 349 | 11.1 s | **41.0%** |

That last column matters for Nov 13: Metrica's own tracking loses the ball on 39–41% of frames,
which is the same order as the ~41% ball-missing rate on this match's graded frames. The
corruption model starts from a dataset that is already noisy in the right way.

## The results sheet

`python -m rondo.harness score` appends one row per (experiment, match) to
`data/external/results.csv`: spellF1 mean and at each IoU, the block-bootstrap interval,
macro-F1 and per-class F1 (including the D class), predicted vs. true possession counts, and
median/p90 boundary error in frames.

**Decision rule.** A module or parameter is kept only when it improves mean held-out spellF1 by
more than the bootstrap noise band — `possession_metrics.SPELL_NOISE_BAND`, 0.04. That is
`score.keep(before, after)`, and it is not a judgement call.

### Reference predictors

Not models — they exist so the sheet records what a floor and a near-ceiling look like before
the decoder can run on external data, and because a first real number is unreadable without
them.

| Experiment | Sample_Game_1 | Sample_Game_2 |
|---|---|---|
| `constant_home` (everything H) | 0.000 | 0.000 |
| `truth` | 1.000 | 1.000 |
| `truth_lag_12` (every boundary half a second late) | 0.813 | 0.767 |
| `truth_lag_25` (one second late) | 0.644 | 0.599 |
| `truth_lag_50` (two seconds late) | 0.464 | 0.444 |

**Read the lag rows before building anything else.** Perfect labels with every boundary half a
second late lose ~0.2 spellF1; a full second late loses ~0.37. Timing is not a polish step on
this metric — it is most of the score, which is the case for boundary snapping (Nov 14) stated
in numbers rather than intuition. The bootstrap interval on these matches is about ±0.035, so a
0.04 noise band is the right threshold to argue with.
