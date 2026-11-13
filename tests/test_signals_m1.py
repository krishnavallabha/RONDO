import unittest

import numpy as np

from rondo.signals import m1_ingest as m1

PITCH = (105.3, 68.0)


def track(n, start=(50.0, 30.0), step=(0.2, 0.0)):
    """A ball rolling in a straight line at `step` metres per frame."""
    i = np.arange(n)[:, None]
    return np.array(start) + i * np.array(step)


class OutOfPitchTest(unittest.TestCase):
    def test_far_outside_rejected_slightly_outside_kept(self):
        xy = track(5)
        xy[1] = (50.0, 100.0)   # 32 m past the touchline: tracking junk
        xy[3] = (50.0, 69.5)    # 1.5 m past the touchline: a real ball out of play
        out, n = m1.reject_out_of_pitch(xy, PITCH)
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(out[1]).all())
        self.assertFalse(np.isnan(out[3]).any())


class TeleportTest(unittest.TestCase):
    def test_lone_spike_removed_and_next_frame_kept(self):
        xy = track(10)
        xy[4] = (80.0, 10.0)
        out, n = m1.reject_teleports(xy)
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(out[4]).all())
        self.assertFalse(np.isnan(out[5]).any())

    def test_bad_first_point_dropped_and_rest_kept(self):
        xy = track(60)
        xy[0] = (5.0, 60.0)
        out, n = m1.reject_teleports(xy)
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(out[0]).all())
        self.assertFalse(np.isnan(out[1:]).any())

    def test_excursion_that_returns_is_dropped(self):
        xy = track(100)
        xy[40:50] = track(10, start=(50.0, 60.0))   # tracker followed something else for 10 frames
        out, n = m1.reject_teleports(xy)
        self.assertEqual(n, 10)
        self.assertTrue(np.isnan(out[40:50]).all())
        self.assertFalse(np.isnan(out[:40]).any() or np.isnan(out[50:]).any())

    def test_lasting_relocation_is_kept_with_a_break(self):
        xy = np.concatenate([track(50), track(50, start=(80.0, 10.0))])
        out, n = m1.reject_teleports(xy)
        self.assertEqual(n, 1)                      # only the first frame after the jump
        self.assertTrue(np.isnan(out[50]).all())
        self.assertFalse(np.isnan(out[:50]).any() or np.isnan(out[51:]).any())

    def test_long_stretch_between_two_relocations_is_kept(self):
        # Ball relocated twice (two dead-ball restarts). The middle 300 frames are real play: far
        # longer than any tracker glitch, so they must survive even though the track "reconnects"
        # around them in the elapsed-time sense. Before the excursion cap they were dropped.
        xy = np.concatenate([track(400), track(300, start=(20.0, 50.0)),
                             track(400, start=(60.0, 30.0))])
        out, n = m1.reject_teleports(xy)
        self.assertEqual(n, 2)  # only the first frame after each jump
        self.assertFalse(np.isnan(out[401:700]).any())

    def test_fast_but_possible_kick_kept(self):
        xy = track(10, step=(1.2, 0.0))  # 30 m/s
        out, n = m1.reject_teleports(xy)
        self.assertEqual(n, 0)
        self.assertFalse(np.isnan(out).any())


class BridgeTest(unittest.TestCase):
    def test_short_gap_interpolated_linearly(self):
        xy = track(30)
        want = xy[10:15].copy()
        xy[10:15] = np.nan
        out, n_short, n_long = m1.bridge_gaps(xy)
        self.assertEqual((n_short, n_long), (5, 0))
        np.testing.assert_allclose(out[10:15], want)

    def test_long_gap_bridged_only_when_trajectory_continues(self):
        xy = track(100)
        xy[30:60] = np.nan                       # 30 frames: needs velocity agreement
        out, n_short, n_long = m1.bridge_gaps(xy)
        self.assertEqual((n_short, n_long), (0, 30))
        self.assertFalse(np.isnan(out).any())

    def test_long_gap_left_open_when_direction_reverses(self):
        xy = track(100)
        xy[60:] = track(40, start=xy[59], step=(-0.2, 0.0)) + np.array([0.0, 0.0])
        xy[30:60] = np.nan
        out, _, n_long = m1.bridge_gaps(xy)
        self.assertEqual(n_long, 0)
        self.assertTrue(np.isnan(out[30:60]).all())

    def test_gap_over_limit_edges_and_impossible_speed_not_bridged(self):
        xy = track(200)
        xy[20:80] = np.nan                       # 60 frames: over the 50-frame limit
        xy[:3] = np.nan                          # leading gap: nothing before it
        out, _, _ = m1.bridge_gaps(xy)
        self.assertTrue(np.isnan(out[20:80]).all())
        self.assertTrue(np.isnan(out[:3]).all())
        far = track(30)
        far[10:13] = np.nan
        far[13:] += np.array([60.0, 0.0])        # would need 60 m in 4 frames
        out, n_short, _ = m1.bridge_gaps(far)
        self.assertEqual(n_short, 0)


class SmoothTest(unittest.TestCase):
    def test_gaps_stay_nan_and_values_do_not_leak_across(self):
        xy = np.concatenate([track(60), np.full((20, 2), np.nan),
                             track(60, start=(5.0, 5.0), step=(0.0, 0.0))])
        out = m1.smooth_runs(xy)
        self.assertTrue(np.isnan(out[60:80]).all())
        np.testing.assert_allclose(out[80:], 5.0, atol=1e-6)

    def test_no_lag_on_a_ramp(self):
        xy = track(120)
        out = m1.smooth_runs(xy)
        np.testing.assert_allclose(out[20:100], xy[20:100], atol=0.05)

    def test_short_runs_left_raw(self):
        xy = track(10)
        np.testing.assert_array_equal(m1.smooth_runs(xy), xy)


class CleanBallTest(unittest.TestCase):
    def test_pipeline_shapes_and_stats(self):
        xy = track(200)
        xy[50] = (50.0, 100.0)
        xy[80] = (95.0, 20.0)
        xy[120:125] = np.nan
        out, stats = m1.clean_ball(xy, PITCH)
        self.assertEqual(out.shape, xy.shape)
        self.assertEqual(stats["out_of_pitch"], 1)
        self.assertEqual(stats["teleports"], 1)
        self.assertEqual(stats["bridged_short"], 7)
        self.assertEqual(int(np.isnan(out).any(axis=1).sum()), 0)

    def test_speed_is_metres_per_second_from_cleaned_positions(self):
        speed = m1.ball_speed_from_positions(track(10, step=(0.2, 0.0)))
        self.assertTrue(np.isnan(speed[0]))
        np.testing.assert_allclose(speed[1:], 5.0)


if __name__ == "__main__":
    unittest.main()
