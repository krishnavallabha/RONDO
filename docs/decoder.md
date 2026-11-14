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

## Decisions (external harness, `python -m rondo.m3_eval`)

Chosen on the 2 Metrica matches, confirmed on the 7 IDSSE matches. Ball conditions: `deadmiss`
(stoppages hidden, as the provider's tracking loses the ball) and `noisy` (deadmiss plus 1.6 m slow
player-position noise, which reproduces our match's ball-to-player scatter).

| Change | Metrica noisy | IDSSE noisy | IDSSE deadmiss | Verdict |
|---|---|---|---|---|
| `min_alive` 10 → 75 | +0.029 | +0.016 | −0.019 | **kept**: helps in the condition that matches our ball, never below −0.02 |
| `min_dead` 25 → 50 (with 75) | +0.005 | −0.002 | +0.001 | not kept: no effect |
| Expected-F1 (MBR) decoding, 40 posterior samples | −0.030 | −0.020 | −0.031 | **dropped**: the heuristic posterior is not calibrated, so samples add noise. Code kept, off by default (`mbr_half`); retest once M7 probabilities are in |
| Restart pull (`restart_weight`) | — | — | — | cannot be tested externally (no Smart Tagging there); acts on ~1,050 frames of our match |

The largest loss in the table is not a decoder setting. Adding 1.6 m of positional noise to the
external matches costs about 0.20 spellF1 (IDSSE 0.53 → 0.33). Better evidence (video ball fill,
learned emissions trained under that noise) is where the remaining score is.
