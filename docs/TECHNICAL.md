# RONDO: frame-by-frame possession for Barcelona vs Athletic Club

**Team:** The Misfits · **Repository:** https://github.com/krishnavallabha/RONDO ·
**Submission:** `The Misfits.csv`, SHA-256 `744dd93f207172cc33c6f8f657abb0a4a31ff9dd9f9cb25b87e7da4cd95d5f46`
(M8 emissions). The earlier M7 submission `cf22e581…` is reproduced by `--model m7`.

## 1. Approach

The ranking metric, spellF1, scores whole possessions matched by overlap, not frames. A per-frame
classifier that flickers on noise creates dozens of false possessions, so we split the problem in
two:

1. **Per-frame evidence (signals).** From the provided tracking and Smart Tagging we compute, for
   every frame, two learned probabilities: `p_dead` (the ball is out of play) and `p_home`
   (Barcelona has the ball, given it is alive).
2. **Sequence decoding (decoder).** An explicit-duration hidden Markov model over H / A / D,
   solved offline with Viterbi, one half at a time. Each state is a chain of sub-states, so a
   possession lasts at least 3 s (`min_alive` 75 frames) and a stoppage at least 1 s
   (`min_dead` 25). Direct H↔A switches (interceptions) are allowed but cost more than going
   through D.

Why this rather than something else: our ball is missing on 41% of graded frames, mostly in long
gaps, so frame-level cues are often absent. A sequence model can carry a possession across a gap
and needs only weak evidence to place a stoppage. Keeping evidence and decoding apart also let us
tune each against labelled external matches separately.

```
tracking (ATD) + Smart Tagging → M1 clean ball → M2 evidence → M8 p_dead, p_home → M3 HMM → CSV
```

## 2. Assets used, and how

| Asset | Used for |
|---|---|
| Tactical tracking (ATD) | Everything. Ball cleaned by fixed physical rules (off-pitch, jumps faster than 40 m/s, short excursions, gap interpolation, 1.25 Hz smoothing). |
| Smart Tagging | Set-piece tags mark restarts. The restart frame is localised from tracking and the taking team pulls the team share (`restart_weight` 0.45, ±10 frames). Kickoff owners come from Law 8 positions. |
| Homography, broadcast video | Not used in the submission (see §6). |
| `games.py` | Pitch size and half boundaries. |

**Learned emissions (M7).** Two logistic regressions in plain numpy:

- **Dead model** (54 features): whether the ball is missing and for how long, position inside the
  gap, player speed divided by this match's median, ball still / out / fast, kicks, distance to
  the lines.
- **Team model** (50 features, no intercept): soft ball-to-team proximity over windows from 1
  frame to 12 s, sustained control, touches as past and future decays, team width. Every feature
  is home minus away and every row is trained from both viewpoints, so swapping the teams gives
  exactly `1 − p_home`.

**M8 stoppage model (submitted; `rondo/signals/m8_stoppage.py`).** On our match the ball is
hidden through almost every long stoppage (98.6% of long slow gaps are already labelled dead), so
the stoppages M7 misses are short ones with the ball in view. M8 keeps M7's team model, retrained
on more conditions, and replaces the dead model with two LightGBM experts (LightGBM 4.6.0,
deterministic, 300 trees): one for frames with the ball seen (M7's dead features plus 28 new ones),
one for ball gaps (only features that do not read the ball). New features: player speed as
within-match quantiles (mean, 90th percentile, share walking) over 0.04 / 3 / 10 s, a sustained
slow-down, ball still for 1.5 s, distance to each line, and gap facts (length, time into it, time
left, where the ball was last and next seen). Features that only identify the data source (number
of tracked players, share standing, ball speed before a gap) are left out: a classifier separated
every external match from ours on them alone.

The extra training conditions come from a simulator that makes an external match's ball look like
ours: each true stoppage is hidden (always if ≥ 10 s, otherwise 29%), the hidden span starts 2.8 s
after the whistle and ends 3.4 s before the restart, plus label-independent dropouts (1.3 per
minute); the ball is low-passed so fewer touches are credited, as on ours. Its parameters were
fitted to our match's label-free statistics (missing share, gap lengths, gaps per minute, in-gap
player speed, credited touches per minute). Three draws are used (`sim0..sim2`).

The decoder divides the class prior out of the classifier posteriors (hybrid HMM/NN scaled
likelihoods), because the transitions already carry it.

## 3. Training data

No label from the provided game is used anywhere. All labels come from other matches:

| Data | Matches | Who labelled it | How we use it |
|---|---|---|---|
| IDSSE Bundesliga open data (Bassek et al. 2024, CC BY 4.0) | 7 | The provider's per-frame `BallStatus` and `BallPossession` channels | Training and held-out validation |
| Metrica Sports sample data, Games 1–2 (CC BY-NC 4.0) | 2 | Metrica's hand-coded events; we derive frame labels with fixed rules | Training and held-out validation |

That is 9 full matches of 25 fps tracking. Provenance, checksums and label rules are in
`docs/EXTERNAL_DATA.md`.

**Making external matches look like ours.** External balls are far cleaner than ours, so every
match is used under five conditions: `full` (unchanged); `deadmiss` (78% of stoppages and 13% of
contacts hidden, as Metrica's tracking loses the ball); `n10dm` and `n16dm` (`deadmiss` plus 1.0 /
1.6 m of slowly drifting player noise); and `n13corrupt` (our match's missing-ball mask
transplanted, plus 1.3 m noise).

**Protocol.** Models are refit out-of-fold, so a match is never scored by a model that saw it
(Metrica held out as a block, IDSSE in three folds). Scoring uses the organisers'
`possession_metrics` through our real decoder. A change is kept only if it helps on both datasets
and does not lose under the two conditions closest to our ball (`n16dm`, `n13corrupt`). The 0.04
noise band is the bar for calling something a clear win.

## 4. Possession rules implemented

| Situation | Rule in RONDO |
|---|---|
| Loose ball, deflection, lone touch | Stays with the last team. Training labels are relabelled so an alive run under 2 s that directly follows the other team's run is merged into it (`m7_learned.controlled`, 50 frames, fixed from the challenge wording and not tuned). The decoder also never starts a possession shorter than 3 s. |
| Contested ball / duel | No change until one team keeps the ball, as above. |
| Change of possession | Needs controlled possession: a sustained run of the other team's evidence. Direct H↔A is allowed (interception) but costs more than going through D. |
| Ball out of play, fouls, stoppages | D. A missing ball with slow players, or last seen out or near a line, is strong dead evidence. |
| Restarts | Possession goes to the taking team: Smart Tagging team tag, or from tracking for throw-ins tagged `N/A`; kickoff owner from Law 8 positions; after a goal, the team that conceded. |
| Outside the halves | X. Each half is decoded alone. |

## 5. Results (held-out external matches)

Mean spellF1 over the five conditions, scored on controlled-possession truth (the challenge's
definition):

| Model | Metrica | IDSSE | Predicted / true possessions |
|---|---|---|---|
| Heuristic evidence only | 0.508 | 0.493 | 0.96 / 0.89 |
| M7, trained on raw provider labels | 0.558 | 0.587 | 1.29 / 1.05 |
| M7, trained on controlled-possession labels | 0.587 | 0.610 | 1.14 / 0.97 |
| **M7, controlled labels + √ possession weights (submitted)** | **0.602** | **0.616** | 0.94 / 0.84 |

Submitted model by condition (full / deadmiss / n10dm / n16dm / n13corrupt): Metrica 0.720 /
0.670 / 0.594 / 0.553 / 0.474; IDSSE 0.722 / 0.670 / 0.627 / 0.569 / 0.492.

**M8.** Leave-one-match-out, scored on the three simulator conditions (the ones made to look like
our match) under a rule fixed in advance (mean gain ≥ 0.02 and above its standard error, ≥ 7 of 9
matches better, none worse by more than 0.03, clean-ball score down by at most 0.01):

| Model | Simulator spellF1 | Metrica | IDSSE | Clean ball | Short-stoppage recall |
|---|---|---|---|---|---|
| M7 (cf22e581) | 0.578 | 0.587 | 0.575 | 0.722 | 0.61 |
| **M8 (submitted)** | **0.605** | **0.614** | **0.602** | **0.728** | **0.82** |

+0.027 ± 0.002, better on 9 of 9 matches (worst +0.016). About 0.020 of it is the dead experts
and 0.007 the team model retrained on the simulator conditions.

**On our match (label-free):** M8 gives 313 possessions (4.5 per live minute); 33.6% dead, 38.7%
home, 27.7% away of graded frames (M7: 326, 4.9, 37.3 / 36.8 / 25.9). 7% of graded frames change.
Our stoppages are shorter than in any external match under both models, so the duration check sits
outside the external range for M7 too.

**Public 5 minutes.** Used only as a correctness check (format, alignment, scorer wiring), never to
choose anything. With 17 possessions its noise band is about ±0.28.

## 6. Tried and dropped

Each was scored on the harness above and kept only if it passed the rule in §3.

| Idea | Result | Why dropped |
|---|---|---|
| Expected-F1 (MBR) decoding, 40 / 100 posterior samples | −0.002 / −0.005 | Neutral once probabilities were calibrated |
| Snapping H↔A switches to the new owner's first touch | ±0.003 | Switch timing is not the loss; missed switches are |
| `min_alive` 2 / 4 / 6 s, `min_dead` 2 s, other switch costs and priors | ≤ ±0.013, mostly 0 | Decoder is saturated |
| Second-stage context model over first-stage outputs (±10 s) | +0.001 / +0.002 on top of √ weights | Not worth a second model |
| 4-state decoder (dead-after-H / dead-after-A) | not built | The team before a stoppage restarts only 43–62% of the time, so the extra state carries no information |
| Equal possession weights in training | −0.011 / −0.016 | Worse |
| Smart Tagging restarts also raise `p_dead` on the 2–5 s before them (`restart_dead`), tested with tags simulated from the truth: 40% of restarts, ±1 s or ±4 s timing error, 20% wrong team | Metrica +0.001, IDSSE −0.003 at ±1 s; Metrica −0.007, IDSSE −0.011 at ±4 s | Worse on IDSSE. The restart pull we ship scores Metrica 0.000, IDSSE +0.001 under the same simulation, so it is harmless |
| Ball-free team-shape features; wider touch or control radius; strict control; player-speed dead ball everywhere | −0.07 to +0.014 | Below the bar or worse on IDSSE |
| 20 SkillCorner open matches added to training | Metrica ±0.003, IDSSE −0.01 | No gain |
| Broadcast-tracking ball fill | fills 6.2% of missing frames; the two balls disagree by > 3 m on 17.6% of shared frames | Too unreliable |
| Broadcast "feed is off the pitch" dead-ball cue | real signal, but no external match to score it on | Unvalidated |
| Ball detection on the video (two public YOLO models) | dead share falls to 22%, outside every external match (32–55%) | Unvalidated; no repository code reads the video |
| M8 decoder re-tune (switch costs, dead prior and bias, emission temperature), nested | +0.000 | Defaults kept |
| Snapping stoppage edges to the ball vanishing / returning | −0.007 to −0.055 | Worse |
| Switch hazard from a learned interception detector | +1.5 points of switch recall within 2 s | Below the +10 points set in advance |
| Ensembles (3 LightGBM seeds; plus M7-style logistic; plus calibration) | +0.000 to +0.001 over M8 | Within noise; not worth three models |

## 7. Limitations, and what we would do with more time

- **No external condition reproduces our ball exactly.** Touches with a player within 1.5 m are
  three times rarer on our match than on any external match, so the harness is a proxy.
- **Interceptions are the main loss.** Only 33–62% of true direct H↔A switches get a predicted
  switch within 2 s. Better ball detection on the video, validated on external video with labels,
  is where we would go next.
- **Smart Tagging cannot be validated directly**: the external matches have no Smart Tagging. We
  simulate it from the truth (§6), with coverage and team error measured on our match, but its
  timing error on our match is unknown.
- **IDSSE splits possession about twice as finely as the target definition**, and 5 of its 7
  matches share a home team. The controlled-possession relabel corrects the first only partly.
- **Throw-ins and fouls** with the ball in view are the hardest stoppages. M8 lifts short-stoppage
  recall from 0.61 to 0.82 on the simulator, but the simulator is still a proxy (a classifier
  tells it from our match with AUC 0.83–0.89).

## 8. Disclosure

- No hand labels and no human-produced data about the provided game. The full-match ground truth
  is never read; the public 5 minutes are a correctness check only.
- Everything learned or tuned is fitted on external matches. Label-free quantities from this
  match's tracking: its missing-ball mask (transplanted into one training condition), its median
  player speed (used as a unit), and for M8's simulator its gap statistics, in-gap player speed
  and credited-touch rate.
- Smart Tagging is used as automatic input, as the FAQ allows.
- AI assistant: only aggregate numbers printed by our own scripts were shared. No match data,
  extract or frame-level file was uploaded to any AI service.
- No match data or frame-level derived data is in the repository (`data/`, `out/` and `*.npz`
  are gitignored). The committed model files (M7 and M8 logistic weights, two LightGBM tree
  files) are fitted on external data only.

## 9. Reproduction

Python 3.12 or 3.13; dependencies pinned in `requirements.txt` (numpy 2.3.2, pandas 2.3.2,
scipy 1.16.2, lightgbm 4.6.0).

```bash
git clone https://github.com/krishnavallabha/RONDO.git && cd RONDO
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/the/unpacked/data/bundle
python -m rondo.signals.export --out out/evidence.npz          # tracking -> evidence, a few min
python -m rondo.run --evidence out/evidence.npz --out "out/The Misfits.csv"
python more-than-a-hack-2027/evaluation/validate_submission.py "out/The Misfits.csv"
```

From a fresh clone this reproduces the submitted file byte for byte (SHA-256 above). Optional:
`python -m rondo.harness fetch && python -m rondo.harness labels` downloads the external data,
`python -m rondo.signals.m7_learned train --relabel 50 --weighting sqrt` refits the M7 weights,
`python -m rondo.signals.m8_stoppage train` refits the M8 model files, and `python -m rondo.m3_eval --relabel 50` reruns the decoder experiments.
`python -m unittest discover -s tests` runs the tests, which need no match data.
