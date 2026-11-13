"""The corruption model: measure what a ball track's noise looks like, then reproduce it.

External matches have a ball the provider tracked almost perfectly. This match's ball is missing
on about 41% of graded frames, in long stretches, and jitters when it is seen. Tuning on a clean
external ball therefore answers a question nobody asked. The corruption model closes that gap,
and it is deliberately split in two:

    measure(ball_xy)   -> a profile: missing rate, gap-length and seen-length distributions,
                          and position jitter. A few hundred bytes of JSON, no data in it.
    apply(ball_xy, p)  -> that profile reproduced on another match's ball.

Why a measured profile rather than transplanting this match's missing mask (which is what
rondo/signals/m2_validate.py does): the mask needs the gated data present, so nobody without it
can validate anything, and CI never can. A profile is measured once, committed under profiles/,
and from then on every external experiment runs from a clean clone with no match data at all.
The transplanted mask stays the stricter check for whoever has the data; this is the portable one.

    python -m rondo.harness noise --evidence out/evidence.npz --out profiles/atd_BAR-ATH.json
    python -m rondo.harness noise --match Sample_Game_1 --out profiles/metrica_native.json
    python -m rondo.harness noise --profile metrica_native        # print one

What a profile does NOT carry: where the gaps are. Only how long they are, how often, and how
far apart. Placement is drawn from a seeded generator, so a corrupted match is reproducible from
the profile and the seed and nothing else.
"""

import json
import os

import numpy as np

# Percentiles stored for each length distribution. Sampling is inverse-CDF interpolation over
# these, which reproduces a long-tailed distribution (gaps run from 1 frame to half a minute)
# far better than a mean and a variance, and costs 21 numbers.
PERCENTILES = tuple(range(0, 101, 5))

# Jitter is estimated from the second difference of a seen stretch: for white noise on a locally
# straight path, var(x[t+1] - 2x[t] + x[t-1]) = 6 sigma^2. The median absolute deviation is used
# rather than the standard deviation so one teleport does not become "jitter".
MAD_TO_SIGMA = 1.4826
SECOND_DIFF_GAIN = np.sqrt(6.0)

LONG_GAP_S = 10.0  # a gap this long is a stoppage, not an occlusion -- reported as a diagnostic

PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiles")


def runs(flags):
    """[(start, stop)] half-open runs of True in a boolean array."""
    flags = np.asarray(flags, dtype=bool)
    edges = np.diff(np.concatenate(([0], flags.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def lengths_of(flags):
    return np.array([stop - start for start, stop in runs(flags)], dtype=np.int64)


def quantiles(values):
    """A length distribution as percentile values, or None when there is nothing to describe."""
    if not len(values):
        return None
    return [float(v) for v in np.percentile(values, PERCENTILES)]


def sample_lengths(table, size, rng):
    """Draw `size` lengths by inverse-CDF interpolation over a stored percentile table."""
    probs = np.array(PERCENTILES, dtype=float) / 100.0
    draws = np.interp(rng.random(size), probs, np.asarray(table, dtype=float))
    return np.maximum(1, np.rint(draws)).astype(np.int64)


def jitter_std(ball_xy, missing):
    """Per-axis position noise in metres, from the second difference of each seen stretch."""
    out = []
    for axis in range(2):
        residuals = []
        for start, stop in runs(~missing):
            track = ball_xy[start:stop, axis]
            if len(track) >= 3:
                residuals.append(np.diff(track, n=2))
        if not residuals:
            out.append(0.0)
            continue
        second = np.concatenate(residuals)
        mad = np.median(np.abs(second - np.median(second)))
        out.append(float(mad * MAD_TO_SIGMA / SECOND_DIFF_GAIN))
    return out


def measure(ball_xy, missing=None, fps=25, source="", scope=""):
    """Measure a noise profile from one ball track. NaN x counts as missing unless told otherwise.

    The three numbers the plan asks for are `missing_rate`, `gap_percentiles` and `jitter_m`.
    `seen_percentiles` comes with them because a missing rate alone does not say whether the ball
    vanishes for a frame at a time or for twenty seconds, and that is the difference between an
    occlusion and a stoppage as far as the decoder is concerned.
    """
    ball_xy = np.asarray(ball_xy, dtype=float)
    if missing is None:
        missing = ~np.isfinite(ball_xy[:, 0])
    missing = np.asarray(missing, dtype=bool)
    gaps, seen = lengths_of(missing), lengths_of(~missing)
    long_frames = int(gaps[gaps >= LONG_GAP_S * fps].sum()) if len(gaps) else 0
    return {
        "source": source,
        "scope": scope,
        "fps": int(fps),
        "n_frames": int(len(missing)),
        "missing_rate": float(missing.mean()) if len(missing) else 0.0,
        "n_gaps": int(len(gaps)),
        "gap_percentiles": quantiles(gaps),
        "seen_percentiles": quantiles(seen),
        "jitter_m": jitter_std(ball_xy, missing),
        # Diagnostics: not used by apply(), compared against it.
        "median_gap_frames": float(np.median(gaps)) if len(gaps) else 0.0,
        "mean_gap_frames": float(gaps.mean()) if len(gaps) else 0.0,
        "max_gap_frames": int(gaps.max()) if len(gaps) else 0,
        "long_gap_share_of_missing": (long_frames / float(missing.sum())
                                      if missing.any() else 0.0),
        "percentiles": list(PERCENTILES),
    }


def gap_mask(n, profile, rng):
    """A missing-ball mask of `n` frames with the profile's gap lengths and missing rate.

    Gap lengths are drawn from the profile until they account for exactly `missing_rate` of the
    frames, and the frames left over are split into that many-plus-one visible runs drawn from
    the profile's seen distribution and scaled to fit. So the rate is exact and the gap-length
    distribution is the measured one, while the *spacing* between gaps is the measured shape
    stretched or squeezed to fit the match.

    That is the deliberate trade-off. Drawing both freely leaves the achieved rate anywhere
    between 34% and 43% on a single match, which would show up as a difference between decoder
    variants scored on different seeds -- a nuisance variable in the one measurement the harness
    exists to make. Gap length and total rate are what the decoder feels; the exact distribution
    of time between gaps is not.
    """
    gap_table, seen_table = profile.get("gap_percentiles"), profile.get("seen_percentiles")
    rate = float(profile.get("missing_rate", 0.0))
    mask = np.zeros(n, dtype=bool)
    target = int(round(rate * n))
    if not gap_table or not seen_table or target < 1:
        return mask

    gaps = []
    total = 0
    while total < target:
        length = int(sample_lengths(gap_table, 1, rng)[0])
        gaps.append(min(length, target - total))
        total += gaps[-1]
    gaps = np.array(gaps, dtype=np.int64)

    seen = sample_lengths(seen_table, len(gaps) + 1, rng).astype(float)
    seen *= (n - target) / max(seen.sum(), 1.0)
    seen = np.maximum(0, np.rint(seen)).astype(np.int64)

    at = 0
    for index, gap in enumerate(gaps):
        at += int(seen[index])
        mask[min(n, at):min(n, at + int(gap))] = True
        at += int(gap)
    return mask


def apply(ball_xy, profile, seed=0, rng=None):
    """`profile`'s missingness and jitter reproduced on another match's ball track.

    Gaps come from gap_mask: measured gap lengths, the measured missing rate, no per-frame coin
    flip -- a Bernoulli mask with the right rate would make 41% missing out of thousands of
    one-frame holes, which is not this match's failure mode and is trivial for M1 to bridge.
    Frames the source match already lacks stay missing, so the result can end up slightly
    noisier than the profile; achieved() says by how much.
    """
    ball_xy = np.asarray(ball_xy, dtype=float)
    rng = rng or np.random.default_rng(seed)
    n = len(ball_xy)
    out = ball_xy.copy()

    out[gap_mask(n, profile, rng)] = np.nan

    jitter = profile.get("jitter_m") or [0.0, 0.0]
    for axis in range(2):
        if jitter[axis] > 0:
            noise = rng.normal(0.0, jitter[axis], n)
            out[:, axis] += noise
    out[~np.isfinite(ball_xy[:, 0])] = np.nan
    return out


def achieved(corrupted, profile, fps=25):
    """What apply() actually produced against what the profile asked for -- the self-check."""
    got = measure(corrupted, fps=fps, source="applied")
    return {
        "missing_rate": (profile["missing_rate"], got["missing_rate"]),
        "median_gap_frames": (profile["median_gap_frames"], got["median_gap_frames"]),
        "mean_gap_frames": (profile["mean_gap_frames"], got["mean_gap_frames"]),
        "long_gap_share_of_missing": (profile["long_gap_share_of_missing"],
                                      got["long_gap_share_of_missing"]),
        "n_gaps": (profile["n_gaps"], got["n_gaps"]),
    }


def path_of(name):
    """profiles/<name>.json for a bare name, or the path as given."""
    if os.path.sep in name or name.endswith(".json"):
        return name
    return os.path.join(PROFILE_DIR, name + ".json")


def save(profile, name):
    target = path_of(name)
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w", newline="\n") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def load(name):
    target = path_of(name)
    if not os.path.isfile(target):
        raise IOError("no noise profile at {}\n\nMeasure one: python -m rondo.harness noise "
                      "--evidence out/evidence.npz --out {}".format(target, target))
    with open(target) as handle:
        return json.load(handle)


def available():
    """Names of the committed profiles."""
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILE_DIR) if f.endswith(".json"))


def from_evidence(path, fps=25):
    """Measure this match's own profile from an M2 evidence file, over the graded halves only.

    Ungraded frames (before kickoff, halftime, after the whistle) have no ball and no score, so
    counting them would inflate the missing rate with frames nobody is asked about.
    """
    from rondo.decoder import graded_blocks
    with np.load(path) as data:
        ball_xy, missing = data["ball_xy"], data["ball_missing"]
    keep = np.zeros(len(missing), dtype=bool)
    for start, stop in graded_blocks():
        keep[start:stop] = True
    return measure(ball_xy[keep], missing[keep], fps,
                   source=os.path.basename(path), scope="graded halves")


def from_match(dataset, match_id, external_dir=None):
    """Measure an external match's own native noise -- the provider's tracking, as it arrived."""
    from rondo.harness import score
    _, truth, meta = score.load(dataset, match_id, external_dir)
    return measure(truth["ball_xy"], truth.get("ball_missing"), meta["fps"],
                   source="{} {}".format(dataset, match_id), scope="whole match")


def from_matches(pairs, external_dir=None, note=""):
    """One profile pooled over several external matches, so the length distributions are not a
    single match's luck. The tracks are concatenated, which adds one artificial run boundary per
    join -- immaterial next to the ~150 gaps each match contributes."""
    from rondo.harness import score
    tracks, flags, fps = [], [], 25
    for dataset, match_id in pairs:
        _, truth, meta = score.load(dataset, match_id, external_dir)
        tracks.append(truth["ball_xy"])
        flags.append(np.asarray(truth["ball_missing"], dtype=bool))
        fps = meta["fps"]
    profile = measure(np.concatenate(tracks), np.concatenate(flags), fps,
                      source=", ".join("{} {}".format(*pair) for pair in pairs),
                      scope="pooled over {} matches".format(len(pairs)))
    if note:
        profile["note"] = note
    return profile
