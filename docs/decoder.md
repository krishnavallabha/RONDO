# Decoder (M3)

`rondo/decoder.py`. Reads the evidence contract (`rondo/contract.py`), writes H/A/D per graded
frame and X outside the two halves.

## Model

- **States** H, A, D, each expanded into a chain of sub-states (explicit-duration HMM), so a spell
  lasts at least `min_alive` (H/A) or `min_dead` (D) frames. A spell may not re-enter its own chain.
- **Decoding**: offline Viterbi per half, using the chain structure (O(sub-states) per frame, about
  3 s per half at 175 sub-states). It returns the same path as dense Viterbi; a test checks this.
- **Transitions**: direct H↔A allowed, but rarer than going through D.
- **Emissions, heuristic path** (default): `dead_ball` remapped so 0.5 ("no idea") weighs the three
  states equally; team share from sustained control, touches, and located restarts.
- **Emissions, M7 path**: when the evidence carries `p_dead` and `p_home` (learned classifier
  posteriors), they replace the heuristics. The class prior (`m7_prior_dead`) is divided out
  because the transitions already carry it (hybrid HMM/NN scaled likelihoods).

## Decisions with M7 emissions (Nov 14 evening, `python -m rondo.m3_eval`)

M7 refit out-of-fold (Metrica as one block, IDSSE in three folds), scored under M7's five
conditions (full, deadmiss, +1.0 m and +1.6 m slow player noise, our missing mask + 1.3 m).
Mean spellF1 over the five; the M7 row reproduces A's out-of-fold numbers.

| Variant | Metrica | IDSSE | pred / true spells (M / I) | Verdict |
|---|---|---|---|---|
| **M7, defaults** | **0.486** | **0.485** | 0.89 / 0.69 | shipped |
| Heuristic emissions | 0.429 | 0.409 | 0.67 / 0.59 | fallback (`export --no-m7`) |
| Expected-F1 (MBR) decoding, 40 / 100 samples | −0.002 / −0.005 | −0.002 / −0.001 | 0.91 / 0.71 | off: neutral now that the probabilities are calibrated, and unchanged from 40 to 100 samples, so the plan's 300 would not change it |
| `min_alive` 50 / 100 / 150 | −0.009 / −0.013 / −0.038 | +0.013 / −0.016 / −0.056 | | 75 stays |
| `min_dead` 50 | 0.000 | 0.000 | | no effect |
| Direct-switch cost 5e-5 / 1e-3 | 0.000 / −0.001 | 0.000 / 0.000 | | no effect |
| `m7_prior_dead` 0.30 / 0.55 | −0.001 / −0.003 | 0.000 / −0.003 | | 0.42 stays |
| Boundary snapping, ±25 / ±50 frames (`snap_switches`) | −0.001 / −0.003 | +0.001 / 0.000 | | off: neutral |

**Boundaries.** Measured as predicted minus true frame, for the nearest boundary of the same kind.
Dead-ball boundaries (alive→D, D→alive) have a median of 0 frames in every condition. With our real
missing mask (`n13corrupt`), only 65–74% fall within 2 s, but those still have a median of 0. A
caveat: `deadmiss` hides the ball exactly at true stoppages, so it flatters these boundaries.
Direct H↔A switches show no bias (median −1 to +3 frames) but an interquartile range of about
±18 frames, and only 33–62% of true switches have a predicted switch within 2 s.
`snap_switches` moves each H↔A switch to the new owner's first touch. It changes nothing, so the
loss is in switches the decoder misses, not in their timing. Code kept, off by default.

No decoder setting moves the score by more than the 0.04 band. The decoder is saturated; what's
left is in the evidence. Under correct (4 s) player noise, 1.6 m costs the heuristic about 0.19 on
IDSSE (deadmiss 0.513 → 0.319) and M7 about 0.09 (0.528 → 0.437). On the harness the decoder
under-splits (0.69–0.89 predicted spells per true spell), while on our match M7 gives more
possessions than the target window suggests. That is a gap between the harness and our match,
not a decoder setting.

## Controlled-possession labels (Nov 16, `python -m rondo.m3_eval --relabel 50`)

CHALLENGE.md: loose or contested ball stays with the last team; a change needs *controlled*
possession ("keeps the ball for a few seconds"). The external labels flip on every touch, so they
teach M7 to split too often. `m7_learned.controlled(gt, 50)` merges a team's alive run shorter
than 2 s (fixed from the wording, not tuned) that directly follows the other team's run; runs after
a stoppage are kept. Both models are scored on the relabelled truth, out-of-fold:

| | Metrica | IDSSE | pred / true spells (M / I) |
|---|---|---|---|
| M7 as shipped (`c6fbd909`) | 0.558 | 0.587 | 1.29 / 1.05 |
| **M7 trained on relabelled labels** | **0.587 (+0.029)** | **0.610 (+0.022)** | 1.14 / 0.97 |
| Heuristic | 0.508 | 0.493 | 0.96 / 0.89 |

Better on 9 of 9 matches and 41 of 45 match-conditions; below the 0.04 band on the mean.
`min_alive` 50 / 100 / 150 on top: no further gain. On our match it cuts possessions from 406 to
375 (6.2 → 5.7 per live minute). Retrain: `python -m rondo.signals.m7_learned train --relabel 50`.

**Shipped (Nov 16):** the committed `m7_weights.json` is now this model; the submission is
`0e6a30c5…` (previous M7 `c6fbd909…`). Chosen on the harness only; on the public window the two
tie (0.353 vs 0.363, both 26 possessions), well inside its noise.

## Last-day experiments (Nov 16, all scored on controlled-possession truth, out-of-fold)

| Change on top of relabelled M7 | Metrica | IDSSE | pred / true spells (M / I) | Verdict |
|---|---|---|---|---|
| **√ training weights** (`--weighting sqrt`: a run of n frames weighs (200/n)^0.5 instead of 200/n) | **+0.015** | **+0.006** | 0.94 / 0.84 | **shipped**: better on 7 of 9 matches, and under both realistic conditions (`n16dm`, `n13corrupt`) on both datasets |
| Equal training weights (`none`) | −0.011 | −0.016 | 0.82 / 0.74 | worse |
| Second-stage context model (first-stage logits over windows up to ±10 s, refit out-of-fold) | −0.022 | −0.008 | | worse alone |
| Second stage on top of √ weights | +0.016 | +0.008 | | +0.001 / +0.002 over √ alone: not worth a second model |
| 4-state decoder (dead-after-H / dead-after-A) | — | — | | not built: on the external matches the team before a stoppage restarts 43–62% of the time (about half), so the extra state carries no information |

Shipped training command: `python -m rondo.signals.m7_learned train --relabel 50 --weighting sqrt`.

## Earlier decisions (heuristic emissions, Nov 14 morning)

Ball conditions: `deadmiss` and `noisy` (deadmiss plus 1.6 m player noise). The noise was
smoothed over 1 s, which made players run at ~14 m/s, so the `noisy` column and the "costs 0.20"
figure in the first version of this page were inflated. The `min_alive` 75 choice still holds
with M7 (table above).

| Change | Metrica noisy | IDSSE noisy | IDSSE deadmiss | Verdict |
|---|---|---|---|---|
| `min_alive` 10 → 75 | +0.029 | +0.016 | −0.019 | **kept**: helps in the condition that matches our ball, never below −0.02 |
| `min_dead` 25 → 50 (with 75) | +0.005 | −0.002 | +0.001 | not kept: no effect |
| Expected-F1 (MBR) decoding, 40 posterior samples | −0.030 | −0.020 | −0.031 | **dropped**: the heuristic posterior is not calibrated, so samples add noise. Code kept, off by default (`mbr_half`); retest once M7 probabilities are in |
| Restart pull (`restart_weight`) | — | — | — | cannot be tested externally (no Smart Tagging there); acts on ~1,050 frames of our match |

