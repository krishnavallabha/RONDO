"""Synthetic tests for the external-match loaders' helpers and the feature experiment."""

import os
import tempfile
import unittest

import numpy as np

import rondo  # noqa: F401  (puts the challenge repo on sys.path)
from rondo.harness import labels
from rondo.signals import external
from rondo.signals import m2_validate as mv

PITCH = (105.0, 68.0)


def two_teams(n, h_x, a_x, h_spread=5.0, a_spread=5.0, players=10):
    """H and A lines of `players` each, centred on x = h_x / a_x, spread along y."""
    y = np.linspace(-1, 1, players)
    xy = np.zeros((n, 2 * players, 2))
    xy[:, :players, 0] = h_x
    xy[:, :players, 1] = 34 + h_spread * y
    xy[:, players:, 0] = a_x
    xy[:, players:, 1] = 34 + a_spread * y
    return xy, np.repeat([0, 1], players).astype(np.int8)


class FeatureTests(unittest.TestCase):

    def test_line_height_positive_when_home_line_is_higher(self):
        # Half 1: H defends x = 0 (mean x 40 < 52.5), A defends x = 105 (mean 70).
        xy, team = two_teams(200, 40.0, 70.0)
        xy[100:, :10, 0] = 45.0  # H pushes up 5 m in the second 100 frames
        half = np.ones(200, dtype=np.int8)
        line = mv.line_height(xy, team, half, PITCH)
        self.assertGreater(np.nanmean(line[150:]), np.nanmean(line[:50]))
        self.assertAlmostEqual(float(line[20]), (40.0 - 35.0) / 10.0)

    def test_marking_gap_positive_when_home_is_freer(self):
        xy, team = two_teams(60, 50.0, 52.0, h_spread=20.0, a_spread=4.0)
        self.assertGreater(np.nanmin(mv.marking_gap(xy, team)), 0)

    def test_too_few_players_is_nan(self):
        xy, team = two_teams(60, 40.0, 70.0)
        xy[:, 3:10] = np.nan  # H down to 3 tracked players
        half = np.ones(60, dtype=np.int8)
        self.assertTrue(np.isnan(mv.line_height(xy, team, half, PITCH)).all())
        self.assertTrue(np.isnan(mv.marking_gap(xy, team)).all())

    def test_centred_removes_each_half_median(self):
        feature = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0, np.nan])
        half = np.array([1, 1, 1, 2, 2, 2, 0], dtype=np.int8)
        out = mv.centred(feature, half)
        np.testing.assert_allclose(out[:6], [-1, 0, 1, -1, 0, 1])
        self.assertTrue(np.isnan(out[6]))


class FitAndDecodeTests(unittest.TestCase):

    def test_fit_recovers_a_known_slope(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20000, 1))
        y = (rng.random(20000) < 1 / (1 + np.exp(-(0.3 + 1.5 * x[:, 0])))).astype(float)
        self.assertAlmostEqual(float(mv.fit_weights(x, y)[0]), 1.5, delta=0.1)

    def test_zero_shift_leaves_the_emission_unchanged(self):
        loglik = np.log(np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]))
        np.testing.assert_allclose(mv.shift_share(loglik, np.zeros(2)), loglik)

    def test_shift_moves_h_a_split_but_not_dead(self):
        loglik = np.log(np.array([[0.4, 0.4, 0.2]]))
        out = np.exp(mv.shift_share(loglik, np.array([np.log(3.0)])))
        self.assertAlmostEqual(out[0, 0] / out[0, 1], 3.0, places=6)
        self.assertAlmostEqual(out[0, 2], 0.2, places=9)
        self.assertAlmostEqual(out[0].sum(), 1.0, places=9)

    def test_decode_keeps_x_outside_halves_and_decodes_each_half(self):
        n = 300
        ev = {"dead_ball": np.full(n, 0.05), "control_H": np.zeros(n, bool),
              "control_A": np.zeros(n, bool), "touch_H": np.zeros(n, bool),
              "touch_A": np.zeros(n, bool)}
        ev["control_H"][:140] = True
        ev["control_A"][160:] = True
        half = np.array([1] * 140 + [0] * 20 + [2] * 140, dtype=np.int8)
        codes = mv.decode_match(ev, half)
        self.assertTrue((codes[140:160] == labels.X).all())
        self.assertEqual(np.bincount(codes[20:140], minlength=4).argmax(), labels.H)
        self.assertEqual(np.bincount(codes[180:], minlength=4).argmax(), labels.A)

    def test_corruption_mask_follows_the_real_pattern(self):
        from rondo import decoder
        real = np.zeros(199319, dtype=bool)
        (a, b), _ = decoder.graded_blocks()
        real[a:a + 10] = True
        mask = mv.corruption_mask(real, 50)
        self.assertTrue(mask[:10].all())
        self.assertFalse(mask[10:].any())


class CorruptionAndPriorTests(unittest.TestCase):

    def test_dead_prior_makes_unknown_uninformative_and_keeps_order(self):
        levels = np.array([0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
        out = mv.dead_prior({"dead_ball": levels})["dead_ball"]
        self.assertAlmostEqual(out[2], 1 / 3)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[-1], 1.0)
        self.assertTrue((np.diff(out) > 0).all())
        # With no control evidence the three states now tie at 1/3 each.
        ev = {"dead_ball": np.array([0.5]), "control_H": np.zeros(1, bool),
              "control_A": np.zeros(1, bool), "touch_H": np.zeros(1, bool),
              "touch_A": np.zeros(1, bool)}
        from rondo import decoder
        np.testing.assert_allclose(np.exp(decoder.emission_loglik(mv.dead_prior(ev)))[0],
                                   [1 / 3] * 3, rtol=1e-5)

    def test_subsample_interpolates_between_kept_frames_and_keeps_gaps(self):
        ball = np.column_stack([np.arange(10.0) ** 2, np.zeros(10)])
        ball[7] = np.nan
        out = mv.subsample(ball, 4)
        self.assertEqual(out[4, 0], 16.0)
        self.assertEqual(out[2, 0], 8.0)  # halfway between frame 0 (0) and frame 4 (16)
        self.assertTrue(np.isnan(out[7]).all())

    def test_occlusion_hides_contacts_and_hits_the_real_share(self):
        n = 2000
        ball = np.column_stack([np.linspace(10, 90, n), np.full(n, 34.0)])
        players = np.full((n, 1, 2), 50.0)
        players[:, 0, 1] = 34.0
        players[:, 0, 0] = 200.0
        players[500:600, 0, 0] = ball[500:600, 0]  # one contact run
        real = np.zeros(199319, dtype=bool)
        (a, b), _ = __import__("rondo").decoder.graded_blocks()
        real[a:a + 400] = True  # the real pattern: 20% of the first 2000 graded frames
        mask = mv.occlusion_mask(ball, players, real, contact_keep=1.0)
        self.assertTrue(mask[494:606].all())
        self.assertAlmostEqual(mask.mean(), 0.2, delta=0.06)
        np.testing.assert_array_equal(mask, mv.occlusion_mask(ball, players, real, 1.0))


class StoppageTests(unittest.TestCase):

    def test_slow_dead_only_touches_unknown_missing_slow_frames(self):
        ev = {"dead_ball": np.array([0.5, 0.5, 0.5, 0.3, 0.5]),
              "ball_missing": np.array([True, True, False, True, True])}
        speed = np.array([0.8, 2.5, 0.8, 0.8, np.nan])
        out = mv.slow_dead(ev, speed, slow=1.2, level=0.8)["dead_ball"]
        np.testing.assert_allclose(out, [0.8, 0.5, 0.5, 0.3, 0.5])
        out = mv.slow_dead(ev, speed, slow=1.2, level=0.8, scope="unknown")["dead_ball"]
        np.testing.assert_allclose(out, [0.8, 0.5, 0.8, 0.3, 0.5])

    def test_deadmiss_hides_stoppages_and_some_play(self):
        n = 3000
        codes = np.full(n, labels.H, dtype=np.int8)
        codes[1000:1500] = labels.D
        codes[2000:2400] = labels.D
        ball = np.column_stack([np.linspace(10, 90, n), np.full(n, 34.0)])
        players = np.full((n, 1, 2), np.nan)
        players[:, 0] = ball + 0.5  # always in contact
        truth = {"ball_xy": ball, "codes": codes}
        out = mv._deadmiss(truth, {}, players, None, dead_rate=1.0, alive_rate=0.1)
        missing = np.isnan(out[:, 0])
        self.assertTrue(missing[codes == labels.D].all())
        self.assertGreater(missing[codes == labels.H].mean(), 0.05)
        again = mv._deadmiss(truth, {}, players, None, dead_rate=1.0, alive_rate=0.1)
        np.testing.assert_array_equal(np.isnan(again), np.isnan(out))


class MetricaLoaderTests(unittest.TestCase):

    def test_metrica_players_scaled_and_row_aligned(self):
        header = (",,,Home,,Home,,,\n,,,1,,2,,,\n"
                  "Period,Frame,Time [s],Player1,,Player2,,Ball,\n")
        rows = "1,1,0.04,0.5,0.5,0.1,0.2,0.5,0.5\n1,3,0.12,0.6,0.5,NaN,NaN,0.5,0.5\n"
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, "metrica", "M")
            os.makedirs(folder)
            for side in ("home", "away"):
                with open(os.path.join(folder, "tracking_{}.csv".format(side)), "w") as handle:
                    handle.write(header.replace("Home", side.title()) + rows)
            original = external.datasets.require
            external.datasets.require = lambda d, m, names, e=None: [
                os.path.join(folder, n + ".csv") for n in names]
            try:
                xy, team = external._metrica("M", root)
            finally:
                external.datasets.require = original
        self.assertEqual(xy.shape, (3, 4, 2))
        np.testing.assert_array_equal(team, [0, 0, 1, 1])
        np.testing.assert_allclose(xy[0, 0], [52.5, 34.0])
        np.testing.assert_allclose(xy[0, 1], [10.5, 13.6])
        self.assertTrue(np.isnan(xy[1]).all())  # frame 2 has no tracking row
        self.assertTrue(np.isnan(xy[2, 1]).all())


if __name__ == "__main__":
    unittest.main()
