# Questions for the organisers

**Status, Nov 12.** The challenge repository says "open an issue" (README §Questions,
RULES §Questions, FAQ), but **Issues are disabled** on
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)
— the GitHub API reports `has_issues: false`, and the issue list is empty, so no issue has been
filed by anyone, by us or by another team. The only published contact is
**hackathon.talentarena@mobileworldcapital.com** (README §Questions, for enrolment and terms of
use). So these go by email until the Issues tab is turned on, and the reply is what we cite in
the technical document.

Rule of this file: **no question here may be answered by looking at the public 5-minute window.**
RULES §2 is explicit that the window is a correctness check and not a tuning set, and the label
conventions below are exactly the kind of thing that would be "tuned" by inspection. We ask
instead and we document the answer.

---

## Owner B — rules and constants

1. **Deadline time zone.** The deadline is Nov 17, 2026. In which time zone does it fall, and is
   it the upload timestamp or the form response that counts?
2. **Kickoff constants.** `games.py` carries both per-half offsets (`1: 1702057, 2: 1702058`) and
   kickoff anchors (`7:27:11`, `75:49:16`). Are the anchors authoritative for anything scored, or
   are they playback conveniences, with the offsets the only alignment that matters?
3. **Baum-Welch on the provided game.** RULES §1 permits "learning without human labels —
   unsupervised or self-supervised methods … provided no person decided any of them". We read
   that as permitting EM / Baum-Welch re-estimation of HMM parameters on this match's *tracking*,
   with no labels of any kind involved. Confirm, since the whole pipeline would be built on it.
4. **Hosted LLM clause.** RULES §Scope allows "LLMs, hosted or local". Does the match data
   licence accepted at enrolment allow match data (tracking, video frames, Smart Tagging) to be
   sent to a hosted model, or is the allowance for code and text only? We currently send **no**
   match data to any hosted service and would rather keep it that way than find out afterwards.

## Owner C — labels, external data, reproducibility

5. **Where a dead ball starts and ends in the ground truth.** For the private ground truth:
   - does `D` start at the frame the ball physically crosses the line, or at the whistle?
   - is the restart kick frame itself `D`, or already `H`/`A` for the restarting team?
   - for a foul, does `D` start at the foul or at the whistle?

   These are the three conventions our external-match labels had to assume (the rules are written
   out in [EXTERNAL_DATA.md](EXTERNAL_DATA.md)). Boundary placement is most of spellF1 — perfect
   labels half a second late score 0.79 — so a systematic convention difference costs more than
   any model choice we could make.

6. **Interceptions.** When possession passes without a stoppage, does the ground truth switch
   `H`→`A` directly at the moment of the opponent's first touch, or is a short `D` inserted?

7. **`X` inside the halves.** Is the private ground truth graded continuously across both halves
   (the spans in `games.py`), or are there stretches inside a half that are `X` — VAR checks,
   ball out of camera, tracking loss? We label every graded frame and never abstain inside a
   half; we want to know whether to expect graded frames the truth itself declines to label.

8. **External data licensing.** We validate on IDSSE/Sportec (CC BY 4.0, 7 Bundesliga matches)
   and the Metrica sample data (CC BY-**NC** 4.0). Is a hackathon entry non-commercial use for
   the purpose of that licence? If not, we drop the Metrica sample and keep IDSSE only.

9. **Reproducibility of external dependencies.** RULES §3 requires the repository to regenerate
   the submission byte for byte. Our external matches are pinned by figshare file id, commit sha,
   md5 and byte size, and re-downloaded on demand rather than committed (the data is too large
   and not ours to redistribute). Is a pinned, checksummed fetch sufficient, or do you expect the
   validation data itself to be reachable in the repository?

---

## Answers

| # | Asked | Answer | Where it landed |
|---|---|---|---|
| | | | |

Fill this in as replies arrive — the technical document's provenance section cites it.
