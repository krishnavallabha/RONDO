"""Harness tests. No network and no downloaded data: every fixture is written here.

The converters are the risk -- a quiet off-by-one or a flipped home/away mapping would poison
every parameter chosen on external matches -- so each one is run over a hand-built match whose
labels are obvious by inspection.
"""

import os
import tempfile
import unittest

import numpy as np

import rondo  # noqa: F401  (puts the challenge repo on sys.path)
import possession_metrics as pm
from rondo.harness import datasets, idsse, labels, metrica, noise, score

HOME_ID, AWAY_ID = "DFL-CLU-HOME", "DFL-CLU-AWAY"

MATCHINFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
  <MatchInformation>
    <General MatchId="TEST01" MatchTitle="Home:Away" Season="2022/2023"
             HomeTeamName="Home" HomeTeamId="{home}"
             GuestTeamName="Away" GuestTeamId="{away}" Result="1:1" />
    <Environment PitchX="105.00" PitchY="68.00" />
  </MatchInformation>
</PutDataRequest>
""".format(home=HOME_ID, away=AWAY_ID)


def positions_xml(rows_by_half, start="2023-05-27T13:30:00.000+00:00"):
    """A positional file in the provider's shape: one player FrameSet, then the ball's.

    rows_by_half maps GameSection -> [(possession, status)], one pair per frame. The player
    FrameSet is there so the test covers the scan skipping ~95% of the file.
    """
    import datetime as dt
    first = dt.datetime.fromisoformat(start)
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest>", "<Positions>"]
    number = 10000
    for section, rows in rows_by_half.items():
        out.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="P1">'.format(
            section, HOME_ID))
        for index in range(len(rows)):
            out.append('<Frame N="{}" X="1.00" Y="2.00" S="0.00" M="1"/>'.format(number + index))
        out.append("</FrameSet>")
        out.append('<FrameSet GameSection="{}" TeamId="BALL" PersonId="BALL">'.format(section))
        for index, (possession, status) in enumerate(rows):
            stamp = (first + dt.timedelta(seconds=(number - 10000 + index) / 25.0)).isoformat()
            out.append('<Frame N="{}" T="{}" X="{:.2f}" Y="-10.00" Z="0.00" S="1.00" M="1"'
                       ' BallPossession="{}" BallStatus="{}"/>'.format(
                           number + index, stamp, index / 10.0, possession, status))
        out.append("</FrameSet>")
        number += len(rows) + 10000
    out += ["</Positions>", "</PutDataRequest>", ""]
    return "\n".join(out)


def events_xml(rows):
    """[(seconds after kickoff, team id)] -> an event file with one Play per entry."""
    import datetime as dt
    first = dt.datetime.fromisoformat("2023-05-27T13:30:00.000+00:00")
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest>"]
    for index, (offset, team) in enumerate(rows):
        when = (first + dt.timedelta(seconds=offset)).isoformat()
        out.append('<Event EventId="{}" EventTime="{}">'.format(index, when))
        out.append('  <Play Team="{}"><Pass/></Play>'.format(team))
        out.append("</Event>")
    out += ["</PutDataRequest>", ""]
    return "\n".join(out)


def write_idsse(folder, rows_by_half, event_rows):
    os.makedirs(folder, exist_ok=True)
    for name, text in (("matchinfo", MATCHINFO),
                       ("positions", positions_xml(rows_by_half)),
                       ("events", events_xml(event_rows))):
        with open(os.path.join(folder, name + ".xml"), "w", newline="\n") as handle:
            handle.write(text)


METRICA_EVENTS = """Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],\
From,To,Start X,Start Y,End X,End Y
Home,SET PIECE,KICK OFF,1,1,0.04,1,0.04,Player1,,0.5,0.5,0.5,0.5
Home,PASS,,1,1,0.04,10,0.4,Player1,Player2,0.5,0.5,0.6,0.5
Home,BALL LOST,INTERCEPTION,1,20,0.8,30,1.2,Player2,,0.6,0.5,0.7,0.5
Away,RECOVERY,INTERCEPTION,1,30,1.2,30,1.2,Player11,,0.7,0.5,0.7,0.5
Away,BALL OUT,,1,40,1.6,45,1.8,Player11,,0.7,0.5,1.0,0.5
Home,SET PIECE,THROW IN,1,60,2.4,60,2.4,Player3,,0.9,0.0,0.9,0.0
Home,PASS,,1,60,2.4,70,2.8,Player3,Player4,0.9,0.0,0.8,0.2
Away,SET PIECE,KICK OFF,2,101,4.04,101,4.04,Player11,,0.5,0.5,0.5,0.5
Away,PASS,,2,101,4.04,110,4.4,Player11,Player12,0.5,0.5,0.4,0.5
"""


def metrica_tracking(n_first, n_second):
    """A tracking CSV in Metrica's three-header-row shape, one player and the ball."""
    lines = [",,,Home,,,", ",,,1,,,", "Period,Frame,Time [s],Player1,,Ball,"]
    for index in range(n_first + n_second):
        period = 1 if index < n_first else 2
        lines.append("{},{},{:.2f},0.5,0.5,0.5,0.5".format(period, index + 1, (index + 1) / 25.0))
    return "\n".join(lines) + "\n"


def write_metrica(folder, n_first=100, n_second=50):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "events.csv"), "w", newline="\n") as handle:
        handle.write(METRICA_EVENTS)
    with open(os.path.join(folder, "tracking_home.csv"), "w", newline="\n") as handle:
        handle.write(metrica_tracking(n_first, n_second))


class RegistryTest(unittest.TestCase):
    def test_seven_idsse_and_two_metrica(self):
        self.assertEqual(len(datasets.match_ids("idsse")), 7)
        self.assertEqual(len(datasets.match_ids("metrica")), 2)
        self.assertEqual(len(datasets.every_match()), 9)

    def test_every_asset_is_pinned(self):
        for dataset, match_id in datasets.every_match():
            for name, (url, md5, size) in datasets.assets(dataset, match_id).items():
                self.assertTrue(url.startswith("https://"), (dataset, match_id, name))
                self.assertEqual(len(md5), 32, (dataset, match_id, name))
                self.assertGreater(size, 0)

    def test_missing_data_says_how_to_get_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(datasets.have("idsse", "J03WMX", external_dir=tmp))
            with self.assertRaisesRegex(IOError, "rondo.harness fetch"):
                datasets.require("idsse", "J03WMX", ("positions",), tmp)


class IdsseTest(unittest.TestCase):
    # First half: 10 frames of home, 10 dead, 10 away. Second half: 10 away, 10 dead.
    FIRST = [(1, 1)] * 10 + [(1, 0)] * 10 + [(2, 1)] * 10
    SECOND = [(2, 1)] * 10 + [(2, 0)] * 10

    def build(self, tmp, first=None, second=None, events=None, **kwargs):
        write_idsse(os.path.join(tmp, "idsse", "TEST01"),
                    {"firstHalf": first or self.FIRST, "secondHalf": second or self.SECOND},
                    events if events is not None else [(0.1, HOME_ID), (1.0, AWAY_ID)])
        return idsse.build("TEST01", tmp, **kwargs)

    def test_reads_provider_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            codes, truth, meta = self.build(tmp)
        self.assertEqual(len(codes), 50)
        np.testing.assert_array_equal(codes[:10], labels.H)
        np.testing.assert_array_equal(codes[10:20], labels.D)
        np.testing.assert_array_equal(codes[20:40], labels.A)
        np.testing.assert_array_equal(codes[40:], labels.D)
        # Halves are concatenated in play order, and the provider's numbering is kept.
        np.testing.assert_array_equal(truth["half"], np.repeat([1, 2], [30, 20]))
        self.assertEqual(truth["frame_n"][0], 10000)
        self.assertEqual(truth["frame_n"][30], 20030)  # the second half's own numbering
        # Centred provider metres -> the 0..105 x 0..68 corner origin RONDO uses.
        self.assertAlmostEqual(truth["ball_xy"][0, 0], 52.5)
        self.assertAlmostEqual(truth["ball_xy"][0, 1], 24.0)
        self.assertEqual(meta["home_team"], "Home")
        self.assertEqual(meta["n_frames"], 50)

    def test_rejects_a_flipped_home_away_mapping(self):
        # Every event says the other team has it: exactly what a flipped mapping looks like.
        flipped = [(0.1, AWAY_ID), (0.2, AWAY_ID), (0.3, AWAY_ID), (1.0, HOME_ID)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "1=home"):
                self.build(tmp, events=flipped)

    def test_dead_frames_do_not_count_toward_agreement(self):
        # The only event lands inside the stoppage, so nothing is checkable and the build stands.
        with tempfile.TemporaryDirectory() as tmp:
            _, _, meta = self.build(tmp, events=[(0.5, AWAY_ID)])
        self.assertEqual(meta["event_agreement_n"], 0)


class MetricaTest(unittest.TestCase):
    def build(self, tmp, **kwargs):
        write_metrica(os.path.join(tmp, "metrica", "Sample_Game_1"), **kwargs)
        return metrica.build("Sample_Game_1", tmp)

    def test_last_touch_and_restart_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            codes, truth, meta = self.build(tmp)
        # Kick-off pass by Home holds until Away's recovery at frame 30 (1-based) -> index 29.
        np.testing.assert_array_equal(codes[:29], labels.H)
        # Away keeps it while the ball runs out of play: dead only from the BALL OUT end frame.
        np.testing.assert_array_equal(codes[29:44], labels.A)
        # Dead from the ball crossing the line (frame 45) to the throw-in kick (frame 60).
        np.testing.assert_array_equal(codes[44:59], labels.D)
        np.testing.assert_array_equal(codes[59:100], labels.H)
        # Second half kicks off with Away, and nothing leaks across the period boundary.
        np.testing.assert_array_equal(codes[100:], labels.A)
        self.assertEqual(meta["n_frames"], 150)
        self.assertEqual(truth["ball_xy"][0, 0], 0.5 * 105.0)

    def test_untracked_frames_are_x(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "metrica", "Sample_Game_1")
            write_metrica(folder)
            # Drop the first 10 rows: frames 1-10 now have no tracking at all.
            path = os.path.join(folder, "tracking_home.csv")
            with open(path) as handle:
                lines = handle.readlines()
            with open(path, "w", newline="\n") as handle:
                handle.writelines(lines[:3] + lines[13:])
            codes, _, _ = metrica.build("Sample_Game_1", tmp)
        np.testing.assert_array_equal(codes[:10], labels.X)
        np.testing.assert_array_equal(codes[10:29], labels.H)


class LabelFileTest(unittest.TestCase):
    def test_round_trip_is_a_valid_submission(self):
        codes = np.array([labels.H] * 5 + [labels.D] * 5 + [labels.A] * 5, dtype=np.int8)
        with tempfile.TemporaryDirectory() as tmp:
            path = labels.write(tmp, codes, {"ball_xy": np.zeros((15, 2))}, {"dataset": "test"})
            with open(path, "rb") as handle:
                self.assertNotIn(b"\r\n", handle.read())
            np.testing.assert_array_equal(labels.read(tmp), codes)
            self.assertEqual(labels.read_meta(tmp)["dataset"], "test")
            self.assertTrue(os.path.isfile(os.path.join(tmp, "truth.npz")))

    def test_summary_counts_possessions_not_frames(self):
        codes = np.array([labels.H] * 10 + [labels.D] * 10 + [labels.A] * 10, dtype=np.int8)
        summary = labels.summary(codes, fps=25)
        self.assertEqual(summary["n_possessions"], 2)
        self.assertEqual(summary["n_dead_segments"], 1)
        self.assertAlmostEqual(summary["alive_share"], 2 / 3.0, places=4)
        self.assertAlmostEqual(summary["home_share_alive"], 0.5)


class ScoreTest(unittest.TestCase):
    GT = np.array([labels.H] * 100 + [labels.D] * 50 + [labels.A] * 100, dtype=np.int8)

    def test_row_matches_the_official_scorer(self):
        row = score.row("exp", "idsse", "TEST01", self.GT, self.GT.copy(), bootstrap_n=20)
        report = pm.evaluate(self.GT, self.GT.copy())
        self.assertAlmostEqual(row["spell_f1_mean"], report["spell_f1_mean"])
        self.assertAlmostEqual(row["macro_f1"], report["macro_f1"])
        self.assertEqual(row["spell_f1_mean"], 1.0)
        self.assertEqual(row["n_gt_spells"], row["n_pred_spells"])
        self.assertEqual(row["boundary_median_frames"], 0.0)

    def test_length_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "frames"):
            score.row("exp", "idsse", "TEST01", self.GT, self.GT[:-1])

    def test_decision_rule_needs_more_than_bootstrap_noise(self):
        before = [{"spell_f1_mean": 0.40}]
        self.assertEqual(score.keep(before, [{"spell_f1_mean": 0.43}])[0], False)
        kept, delta = score.keep(before, [{"spell_f1_mean": 0.50}])
        self.assertTrue(kept)
        self.assertAlmostEqual(delta, 0.10)

    def test_sheet_appends_with_one_header(self):
        rows = [score.row("a", "idsse", "TEST01", self.GT, self.GT.copy())]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.csv")
            score.write_rows(rows, path)
            score.write_rows(rows, path)
            with open(path) as handle:
                lines = handle.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("experiment,dataset,match"))

    def test_baselines_are_scoreable(self):
        gt = self.GT
        floor = score.BASELINES["constant_home"]({}, {}, gt)
        # One spell covering the whole match: on a real match this scores 0.0, and even on a
        # toy with two possessions it cannot get near a real submission.
        self.assertLess(score.row("f", "d", "m", gt, floor)["spell_f1_mean"], 0.3)
        # A lag baseline keeps the possessions and moves only their edges, which is what makes
        # it a calibration of boundary error: every boundary lands 25 frames late.
        lagged = score.BASELINES["truth_lag_25"]({}, {}, gt)
        row = score.row("l", "d", "m", gt, lagged)
        self.assertEqual(len(lagged), len(gt))
        self.assertEqual(row["n_pred_spells"], row["n_gt_spells"])
        self.assertEqual(row["boundary_median_frames"], 25.0)


class NoiseTest(unittest.TestCase):
    """The corruption model. Everything here is synthetic: no data, no network."""

    @staticmethod
    def track(n, gap=200, seen=300):
        """A clean straight ball track with regular gaps -- a profile with known statistics."""
        xy = np.stack([np.linspace(0, 105, n), np.full(n, 34.0)], axis=1)
        missing = np.zeros(n, dtype=bool)
        for start in range(seen, n, seen + gap):
            missing[start:start + gap] = True
        xy[missing] = np.nan
        return xy, missing

    def test_measure_reads_rate_gaps_and_jitter(self):
        xy, missing = self.track(10000)
        profile = noise.measure(xy)
        self.assertAlmostEqual(profile["missing_rate"], missing.mean(), places=6)
        self.assertEqual(profile["median_gap_frames"], 200.0)
        self.assertEqual(profile["max_gap_frames"], 200)
        self.assertEqual(profile["long_gap_share_of_missing"], 0.0)  # 200 frames = 8 s < 10 s
        # A straight track with no noise: jitter must read as zero, not as its curvature.
        self.assertLess(max(profile["jitter_m"]), 1e-9)

    def test_measure_finds_injected_jitter(self):
        n = 40000
        rng = np.random.default_rng(0)
        xy = np.stack([np.linspace(0, 105, n), np.full(n, 34.0)], axis=1)
        xy[:, 0] += rng.normal(0, 0.05, n)
        profile = noise.measure(xy)
        self.assertAlmostEqual(profile["jitter_m"][0], 0.05, places=2)
        self.assertLess(profile["jitter_m"][1], 1e-9)

    def test_apply_reproduces_the_profile_it_was_given(self):
        profile = noise.load("metrica_native")
        clean = np.zeros((400000, 2))
        got = noise.measure(noise.apply(clean, profile, seed=0))
        # Rate and the share of missing frames in long gaps are what the decoder feels; both
        # within a couple of points over a long track.
        self.assertAlmostEqual(got["missing_rate"], profile["missing_rate"], delta=0.005)
        self.assertAlmostEqual(got["long_gap_share_of_missing"],
                               profile["long_gap_share_of_missing"], delta=0.05)

    def test_gaps_are_stretches_not_a_per_frame_coin_flip(self):
        # The failure mode this guards: a Bernoulli mask with the right rate is thousands of
        # one-frame holes, which M1 bridges trivially and which this match does not have.
        profile = noise.load("metrica_native")
        got = noise.measure(noise.apply(np.zeros((200000, 2)), profile, seed=3))
        self.assertGreater(got["median_gap_frames"], 20)
        self.assertLess(got["n_gaps"], 500)

    def test_apply_is_seeded_and_keeps_existing_holes(self):
        profile = noise.load("metrica_native")
        clean = np.zeros((50000, 2))
        clean[:100] = np.nan  # already missing in the source match
        first = noise.apply(clean, profile, seed=7)
        self.assertTrue(np.array_equal(np.isnan(first), np.isnan(noise.apply(clean, profile, 7))))
        self.assertFalse(np.array_equal(np.isnan(first),
                                        np.isnan(noise.apply(clean, profile, 8))))
        self.assertTrue(np.isnan(first[:100, 0]).all())

    def test_profile_round_trips_through_json(self):
        xy, _ = self.track(5000)
        profile = noise.measure(xy, source="test")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.json")
            noise.save(profile, path)
            self.assertEqual(noise.load(path), profile)

    def test_committed_profile_has_what_apply_needs(self):
        profile = noise.load("metrica_native")
        for key in ("missing_rate", "gap_percentiles", "seen_percentiles", "jitter_m", "note"):
            self.assertIn(key, profile)
        self.assertEqual(len(profile["gap_percentiles"]), len(noise.PERCENTILES))
        self.assertGreater(profile["missing_rate"], 0.3)  # this match loses the ball on ~41%

    def test_missing_profile_says_how_to_make_one(self):
        with self.assertRaisesRegex(IOError, "rondo.harness noise"):
            noise.load(os.path.join("profiles", "not_measured_yet.json"))


class RunTest(unittest.TestCase):
    def test_decode_commits_inside_halves_and_abstains_outside(self):
        from rondo.harness import run as harness_run
        from rondo import contract
        n = 600
        half = np.zeros(n, dtype=np.int8)
        half[100:300] = 1
        half[400:600] = 2
        ev = contract.empty(n)
        ev["control_H"][100:300] = True
        ev["control_A"][400:600] = True
        codes = harness_run.decode(ev, half)
        self.assertTrue((codes[:100] == labels.X).all())
        self.assertTrue((codes[300:400] == labels.X).all())
        self.assertTrue((codes[100:300] == labels.H).all())
        self.assertTrue((codes[400:600] == labels.A).all())

    def test_report_states_whether_the_target_is_beaten(self):
        from rondo.harness import run as harness_run
        lines = []
        rows = [{"experiment": "harness_x", "dataset": "idsse", "match": "M",
                 "spell_f1_mean": 0.30},
                {"experiment": "harness_x", "dataset": "idsse", "match": "N",
                 "spell_f1_mean": 0.40}]
        harness_run.report(rows, target=0.2433, log=lines.append)
        self.assertEqual(len(lines), 1)
        self.assertIn("ABOVE", lines[0])
        self.assertIn("0.3500", lines[0])  # the mean, not the best match


if __name__ == "__main__":
    unittest.main()
