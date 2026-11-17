"""Synthetic tests for the M8 stoppage model (no match data needed)."""

import os
import unittest

import numpy as np

import rondo  # noqa: F401  (puts the challenge repo on sys.path)
from rondo.signals import m2_evidence as m2
from rondo.signals import m7_learned as m7
from rondo.signals import m8_stoppage as m8

PITCH = (105.0, 68.0)


def toy_match(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    y = np.linspace(8, 60, 10)
    players = np.zeros((n, 20, 2))
    players[:, :10, 0] = 40.0
    players[:, :10, 1] = y
    players[:, 10:, 0] = 60.0
    players[:, 10:, 1] = y
    players += np.cumsum(rng.normal(0, 0.03, players.shape), axis=0)
    team = np.repeat([0, 1], 10).astype(np.int8)
    t = np.arange(n)
    ball = np.column_stack([np.full(n, 40.5), 34 + 20 * np.sin(2 * np.pi * t / 250.0)])
    ball[900:1100] = np.nan
    return ball, players, team, np.ones(n, dtype=np.int8)


class FeatureTests(unittest.TestCase):

    def setUp(self):
        ball, self.players, self.team, self.half = toy_match()
        self.ev, _ = m2.build_evidence(ball, self.players, self.team, PITCH)

    def test_stoppage_features_shape_and_ball_columns_absent_in_gaps(self):
        XS, names = m8.stoppage_features(self.ev, self.players, self.half, PITCH)
        self.assertEqual(XS.shape, (len(self.half), len(names)))
        self.assertEqual(len(names), 33)
        miss = ~np.isfinite(self.ev["ball_xy"][:, 0])
        for name in m8.BALL_XS[:5]:
            self.assertTrue(np.isnan(XS[miss, names.index(name)]).all(), name)
        self.assertTrue(np.isfinite(XS[miss, names.index("gap_len")]).all())

    def test_expert_columns_drop_identity_and_ball_features(self):
        _, _, _, names_dead = m7.features(self.ev, self.players, self.team, self.half)
        _, names_stop = m8.stoppage_features(self.ev, self.players, self.half, PITCH)
        vis_xs, gap_xd, gap_xs = m8.expert_columns(names_dead, names_stop)
        self.assertFalse({names_stop[i] for i in vis_xs} & set(m8.IDENTITY))
        self.assertFalse({names_stop[i] for i in gap_xs} & (set(m8.IDENTITY) | set(m8.BALL_XS)))
        self.assertFalse({names_dead[i] for i in gap_xd} & set(m8.BALL_XD))


class SimulatorTests(unittest.TestCase):

    def test_long_stoppage_is_hidden_with_offset_edges(self):
        gt = np.zeros(6000, np.int8)
        gt[2000:3000] = 2                            # a 40 s stoppage
        half = np.ones(6000, np.int8)
        p = dict(m8.sim_params(0), drop_rate=0.0)
        mask = m8.sim_mask(gt, half, p, np.random.default_rng(0))
        gaps = m8.runs(mask)
        self.assertEqual(len(gaps), 1)
        a, b = gaps[0]
        self.assertAlmostEqual((a - 2000) / 25, p["mu_s"], delta=4)
        self.assertAlmostEqual((b - 3000) / 25, p["mu_e"], delta=4)

    def test_ball_smooth_keeps_gaps(self):
        ball, _, _, _ = toy_match()
        out = m8.ball_smooth(ball, 6.0)
        self.assertTrue(np.array_equal(np.isnan(out), np.isnan(ball)))

    def test_sim_params_are_fixed(self):
        self.assertEqual(m8.sim_params(1), m8.sim_params(1))
        self.assertEqual(m8.sim_params(0)["sigma_b"], m8.SIGMA_B)


@unittest.skipUnless(all(os.path.isfile(p) for p in m8.EXPERT_FILES + (m8.META, m8.TEAM_WEIGHTS)),
                     "M8 model files not trained")
class ModelFileTests(unittest.TestCase):

    def test_probabilities_are_valid_and_team_flip_is_exact(self):
        ball, players, team, half = toy_match()
        ev, _ = m2.build_evidence(ball, players, team, PITCH)
        p_dead, p_home = m8.probabilities(ev, players, team, half, PITCH)
        self.assertTrue(((p_dead > 0) & (p_dead < 1)).all())
        swapped = (1 - team).astype(np.int8)
        ev2, _ = m2.build_evidence(ball, players, swapped, PITCH)
        _, p_home2 = m8.probabilities(ev2, players, swapped, half, PITCH)
        np.testing.assert_allclose(p_home, 1 - p_home2, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
