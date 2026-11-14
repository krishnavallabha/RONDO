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
| Expected-F1 (MBR) decoding | −0.002 | −0.002 | 0.91 / 0.71 | off: neutral now that the probabilities are calibrated |
| `min_alive` 50 / 100 / 150 | −0.009 / −0.013 / −0.038 | +0.013 / −0.016 / −0.056 | | 75 stays |
| `min_dead` 50 | 0.000 | 0.000 | | no effect |
| Direct-switch cost 5e-5 / 1e-3 | 0.000 / −0.001 | 0.000 / 0.000 | | no effect |
| `m7_prior_dead` 0.30 / 0.55 | −0.001 / −0.003 | 0.000 / −0.003 | | 0.42 stays |

No decoder setting moves the score by more than the 0.04 band. The decoder is saturated; what's
left is in the evidence. Under correct (4 s) player noise, 1.6 m costs the heuristic about 0.19 on
IDSSE (deadmiss 0.513 → 0.319) and M7 about 0.09 (0.528 → 0.437). On the harness the decoder
under-splits (0.69–0.89 predicted spells per true spell), while on our match M7 gives more
possessions than the target window suggests. That is a gap between the harness and our match,
not a decoder setting.

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

