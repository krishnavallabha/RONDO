import os
import tempfile
import unittest

import numpy as np

import rondo  # noqa: F401  (puts the challenge repo on sys.path)
import possession_metrics as pm
from games import GAME, GAMES
from rondo import contract, run
from rondo.signals import export
from rondo.signals import m2_evidence as m2

N = GAMES[GAME]["n_video_frames"]
PITCH = GAMES[GAME]["pitch"]


def kicked_ball(n=100, turn_at=50):
    """Rolls along +x at 5 m/s, then is kicked hard along -y at frame `turn_at`."""
    xy = np.zeros((n, 2))
    xy[:turn_at + 1, 0] = 30.0 + 0.2 * np.arange(turn_at + 1)
    xy[turn_at + 1:, 0] = xy[turn_at, 0]
    xy[:, 1] = 30.0
    xy[turn_at + 1:, 1] = 30.0 - 0.8 * np.arange(1, n - turn_at)
    return xy


def nearest_at(n, frames, d_home, d_away):
    near = np.full((n, 2), np.nan)
    near[frames] = (d_home, d_away)
    return near


class TouchTest(unittest.TestCase):
    def test_touch_goes_to_the_nearer_team(self):
        xy = kicked_ball()
        tH, tA = m2.detect_touches(xy, nearest_at(100, slice(49, 52), 0.5, 1.2))
        self.assertEqual(np.flatnonzero(tH).tolist(), [50])
        self.assertFalse(tA.any())
        tH, tA = m2.detect_touches(xy, nearest_at(100, slice(49, 52), 1.3, 0.4))
        self.assertFalse(tH.any())
        self.assertEqual(np.flatnonzero(tA).tolist(), [50])

    def test_near_tie_credits_both(self):
        tH, tA = m2.detect_touches(kicked_ball(), nearest_at(100, slice(49, 52), 0.9, 1.1))
        self.assertTrue(tH[50] and tA[50])

    def test_nobody_close_no_touch(self):
        tH, tA = m2.detect_touches(kicked_ball(), nearest_at(100, slice(49, 52), 4.0, 5.0))
        self.assertFalse(tH.any() or tA.any())

    def test_touch_found_next_to_a_gap(self):
        xy = kicked_ball()
        xy[56:58] = np.nan                     # no ball 6 frames after the kick
        tH, _ = m2.detect_touches(xy, nearest_at(100, slice(49, 52), 0.5, 3.0))
        self.assertTrue(tH[49:52].any())


class DeadBallTest(unittest.TestCase):
    def test_last_seen_out_expires_after_stale_window(self):
        n = 1200
        xy = np.full((n, 2), np.nan)
        xy[:100] = np.column_stack([30.0 + 0.2 * np.arange(100), np.full(100, 30.0)])
        xy[100:110] = (50.0, -3.0)             # out over the touchline, then lost
        players = np.full((n, 46, 2), np.nan)
        p = m2.detect_dead_ball(xy, m2.ball_speed_mps(xy), players, PITCH)
        self.assertTrue(np.all(p[100:110] == 1.0))
        self.assertTrue(np.all(p[110:110 + m2.MISSING_STALE_FRAMES] == 0.9))
        self.assertTrue(np.all(p[110 + m2.MISSING_STALE_FRAMES:] == 0.5))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))


def teams_frame(n, spread_home, spread_away, players=11):
    """Two teams of `players` on a circle of the given radius, the same on every frame."""
    ang = np.linspace(0, 2 * np.pi, players, endpoint=False)
    ring = np.column_stack([np.cos(ang), np.sin(ang)])
    xy = np.concatenate([50 + spread_home * ring, 50 + spread_away * ring])
    return np.repeat(xy[None], n, axis=0), np.array([0] * players + [1] * players)


class ShapeTest(unittest.TestCase):
    def test_wider_team_gives_positive_ratio(self):
        xy, team = teams_frame(40, 20.0, 10.0)
        r = m2.team_spread_ratio(xy, team)
        np.testing.assert_allclose(r, np.log(2.0), atol=1e-9)
        xy, team = teams_frame(40, 8.0, 16.0)
        self.assertTrue(np.all(m2.team_spread_ratio(xy, team) < 0))

    def test_too_few_players_is_nan_and_not_smoothed_over(self):
        xy, team = teams_frame(60, 20.0, 10.0)
        xy[20:30, :5] = np.nan                 # home down to 6 tracked players
        r = m2.team_spread_ratio(xy, team)
        self.assertTrue(np.isnan(r[20:30]).all())
        self.assertFalse(np.isnan(r[:20]).any() or np.isnan(r[30:]).any())

    def test_smoothing_averages_a_one_frame_blip(self):
        xy, team = teams_frame(60, 10.0, 10.0)
        wide, _ = teams_frame(1, 40.0, 10.0)
        xy[30] = wide[0]
        raw = m2.team_spread_ratio(xy, team, smooth=1)
        smooth = m2.team_spread_ratio(xy, team)
        self.assertGreater(raw[30], 1.0)
        self.assertLess(smooth[30], raw[30] / 10)


class RestartTest(unittest.TestCase):
    def setUp(self):
        self.n = 200
        self.none = np.zeros(self.n, dtype=bool)

    def test_throw_in_owner_from_taker_at_the_ball(self):
        dead = np.full(self.n, 0.5)
        dead[:50] = 0.9
        near = nearest_at(self.n, slice(60, 70), 2.5, 1.0)
        f, team = m2.localise_restart(0, self.n, dead, self.none, self.none, None, nearest=near)
        self.assertEqual((f, team), (60, m2.AWAY))

    def test_throw_in_unlocated_when_nobody_at_the_ball(self):
        dead = np.full(self.n, 0.5)
        dead[:50] = 0.9
        near = nearest_at(self.n, slice(60, 70), 6.0, 7.0)
        f, team = m2.localise_restart(0, self.n, dead, self.none, self.none, None, nearest=near)
        self.assertEqual((f, team), (None, -1))

    def test_kickoff_frame_from_centre_spot_then_window_middle(self):
        dead = np.full(self.n, 0.5)
        centre = (PITCH[0] / 2, PITCH[1] / 2)
        xy = np.full((self.n, 2), np.nan)
        xy[20:41] = centre
        xy[41:60] = (70.0, 34.0)
        f, team = m2.localise_restart(0, self.n, dead, self.none, self.none, m2.HOME,
                                      ball_xy=xy, centre=centre)
        self.assertEqual((f, team), (40, m2.HOME))
        f, team = m2.localise_restart(0, self.n, dead, self.none, self.none, m2.AWAY,
                                      ball_xy=np.full((self.n, 2), np.nan), centre=centre)
        self.assertEqual((f, team), (self.n // 2, m2.AWAY))


def m2_style_evidence():
    """Evidence as M2 builds it: int8 teams with -1 for unknown."""
    rng = np.random.default_rng(0)
    ev = {
        "ball_xy": np.full((N, 2), np.nan),
        "ball_missing": np.ones(N, dtype=bool),
        "dead_ball": np.full(N, 0.5),
        "smart_tag_dead": np.full(N, 0.5),
        "restart_team": np.full(N, -1, dtype=np.int8),
        "smart_tag_team": np.full(N, -1, dtype=np.int8),
        "shape_spread": np.full(N, np.nan),
    }
    for key in ("touch_H", "touch_A", "control_H", "control_A", "duel_zone"):
        ev[key] = np.zeros(N, dtype=bool)
    ev["control_H"][20000:21000] = True
    ev["control_A"][30000:31000] = True
    ev["restart_team"][11175:11196] = 1
    ev["smart_tag_team"][:] = rng.integers(-1, 2, N).astype(np.int8)
    return ev


class ExportContractTest(unittest.TestCase):
    def test_m2_style_evidence_passes_m2_check(self):
        m2.validate_evidence(m2_style_evidence(), N)

    def test_to_contract_turns_minus_one_into_nan_and_passes_contract(self):
        ev = m2_style_evidence()
        out = contract.check(export.to_contract(ev), N)
        for key in export.TEAM_KEYS:
            self.assertEqual(out[key].dtype, float)
            np.testing.assert_array_equal(np.isnan(out[key]), ev[key] == -1)
            np.testing.assert_array_equal(out[key][ev[key] >= 0], ev[key][ev[key] >= 0])

    def test_raw_m2_teams_rejected_by_contract(self):
        with self.assertRaises(ValueError):
            contract.check(m2_style_evidence(), N)

    def test_saved_evidence_decodes_to_a_valid_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz = os.path.join(tmp, "evidence.npz")
            csv = os.path.join(tmp, "out.csv")
            np.savez_compressed(npz, **export.to_contract(m2_style_evidence()))
            run.main(["--evidence", npz, "--out", csv])
            with np.load(npz) as saved:
                self.assertIn("shape_spread", saved.files)
            self.assertIsNone(pm.validate_submission(pm.read_label_csv(csv), N))


if __name__ == "__main__":
    unittest.main()


def synthetic_match(n=1500, pitch=(105.0, 68.0)):
    """Plain arrays shaped like an external match: 22 players jogging, a ball passed around."""
    rng = np.random.default_rng(1)
    t = np.arange(n)[:, None]
    base = rng.uniform((10, 8), (pitch[0] - 10, pitch[1] - 8), size=(22, 2))
    players = base[None] + 3 * np.stack([np.sin(t / 50 + np.arange(22)), np.cos(t / 60 + np.arange(22))], axis=2)
    ball = np.column_stack([pitch[0] / 2 + 30 * np.sin(np.arange(n) / 80), pitch[1] / 2 + 20 * np.cos(np.arange(n) / 70)])
    ball[400:430] = np.nan
    return ball, players, np.array([0] * 11 + [1] * 11)


class KickoffOwnerTest(unittest.TestCase):
    def lineup(self, kicker):
        """Kicking team: one player on the spot. Opponents: all at least 10 m away."""
        centre = np.array([52.5, 34.0])
        kicking = centre + np.array([[0.5, 0.0]] + [[-15.0 - i, 0.0] for i in range(10)])
        waiting = centre + np.array([[12.0 + i, 0.0] for i in range(11)])
        home, away = (kicking, waiting) if kicker == m2.HOME else (waiting, kicking)
        xy = np.concatenate([home, away])
        return np.repeat(xy[None], 40, axis=0), np.array([0] * 11 + [1] * 11)

    def test_kicking_team_read_from_positions(self):
        for kicker in (m2.HOME, m2.AWAY):
            xy, team = self.lineup(kicker)
            self.assertEqual(m2.kickoff_owner(xy, team, 30, (105.0, 68.0)), kicker)

    def test_unclear_lineup_gives_unknown(self):
        xy, team = self.lineup(m2.HOME)
        xy[:, 11:] = xy[:, :11] + 0.3          # both teams crowd the spot
        self.assertEqual(m2.kickoff_owner(xy, team, 30, (105.0, 68.0)), -1)


class KickoffAlternationTest(unittest.TestCase):
    def test_second_half_owner_follows_from_the_first(self):
        n = 400
        xy, team = KickoffOwnerTest().lineup(m2.AWAY)
        players = np.repeat(xy[:1], n, axis=0)
        players[200:] += np.array([0.0, 30.0])      # second-half lineup unreadable: nobody at the spot
        ball = np.full((n, 2), np.nan)
        ev, info = m2.build_evidence(ball, players, team, (105.0, 68.0), clean=False,
                                     kickoffs={"1st Half": 30, "2nd Half": 300})
        self.assertEqual(info["kickoff_owners"], {"1st Half": m2.AWAY, "2nd Half": m2.HOME})
        self.assertEqual(ev["restart_team"][300], m2.HOME)


class BuildEvidenceTest(unittest.TestCase):
    def test_external_match_gives_contract_valid_evidence(self):
        ball, players, team = synthetic_match()
        ev, info = m2.build_evidence(ball, players, team, pitch=(105.0, 68.0))
        self.assertEqual(set(ev), set(m2.EVIDENCE_SPEC))
        self.assertTrue(all(len(v) == len(ball) for v in ev.values()))
        self.assertTrue((ev["restart_team"] == -1).all())          # no Smart Tagging, no kickoffs
        self.assertTrue((ev["smart_tag_dead"] == 0.5).all())
        self.assertIn("clean_stats", info)
        with tempfile.TemporaryDirectory() as tmp:
            path = export.save(ev, os.path.join(tmp, "ext.npz"))
            with np.load(path) as saved:
                contract.check({k: saved[k] for k in saved.files}, len(ball))

    def test_bad_inputs_are_rejected_with_a_clear_error(self):
        ball, players, team = synthetic_match(300)
        with self.assertRaisesRegex(ValueError, "fps"):
            m2.build_evidence(ball, players, team, (105.0, 68.0), fps=30)
        with self.assertRaisesRegex(ValueError, "outside the pitch"):
            m2.build_evidence(ball - 52.5, players - (52.5, 34.0), team, (105.0, 68.0))
        with self.assertRaisesRegex(ValueError, "only 0"):
            m2.build_evidence(ball, players, team + 1, (105.0, 68.0))
        with self.assertRaisesRegex(ValueError, "expected"):
            m2.build_evidence(ball, players[:, :20], team, (105.0, 68.0))
