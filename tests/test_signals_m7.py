"""Synthetic tests for the M7 learned emissions (no match data needed)."""

import os
import tempfile
import unittest

import numpy as np

import rondo  # noqa: F401  (puts the challenge repo on sys.path)
from rondo import contract
from rondo.signals import m2_evidence as m2
from rondo.signals import m7_learned as m7

PITCH = (105.0, 68.0)


def toy_match(n=1500, seed=0):
    """Two lines of 10 players; the ball is passed among the home line, then lost for 8 s."""
    rng = np.random.default_rng(seed)
    y = np.linspace(8, 60, 10)
    players = np.zeros((n, 20, 2))
    players[:, :10, 0] = 40.0
    players[:, :10, 1] = y
    players[:, 10:, 0] = 60.0
    players[:, 10:, 1] = y
    players += np.cumsum(rng.normal(0, 0.03, players.shape), axis=0)   # everyone drifts a little
    team = np.repeat([0, 1], 10).astype(np.int8)
    t = np.arange(n)
    ball = np.column_stack([np.full(n, 40.5), 34 + 20 * np.sin(2 * np.pi * t / 250.0)])
    ball[900:1100] = np.nan
    half = np.ones(n, dtype=np.int8)
    return ball, players, team, half


class HelperTests(unittest.TestCase):

    def test_rmean_is_a_centred_average_with_shrinking_edges(self):
        x = np.arange(10, dtype=float)
        out = m7.rmean(x, 3)
        self.assertAlmostEqual(out[5], 5.0)
        self.assertAlmostEqual(out[0], 0.5)          # only frames 0 and 1 are inside the window
        self.assertIs(m7.rmean(x, 1), x)

    def test_run_len_gives_every_frame_its_run_length(self):
        mask = np.array([0, 1, 1, 1, 0, 1, 0], dtype=bool)
        self.assertEqual(m7.run_len(mask).tolist(), [0, 3, 3, 3, 0, 1, 0])

    def test_since_counts_frames_to_the_previous_and_next_true(self):
        prev, nxt = m7.since(np.array([0, 1, 0, 0, 1, 0], dtype=bool))
        self.assertEqual(prev.tolist(), [1e4, 0, 1, 2, 0, 1])
        self.assertEqual(nxt.tolist(), [1, 3, 2, 1, 1e4, 1e4])
        prev, nxt = m7.since(np.zeros(3, dtype=bool))
        self.assertTrue((prev == 1e4).all() and (nxt == 1e4).all())

    def test_decay_looks_back_or_forward(self):
        x = np.zeros(100)
        x[50] = 1.0
        past, future = m7.decay(x, 1.0, True), m7.decay(x, 1.0, False)
        self.assertEqual(past[49], 0.0)
        self.assertAlmostEqual(past[50], 1.0)
        self.assertAlmostEqual(past[75], np.exp(-1.0))
        self.assertEqual(future[51], 0.0)
        self.assertAlmostEqual(future[25], np.exp(-1.0))

    def test_slow_noise_is_seeded_and_scaled(self):
        a, b = m7.slow_noise(5000, 1.6, 3), m7.slow_noise(5000, 1.6, 3)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.allclose(a.std(0), 1.6))
        self.assertLess(np.abs(np.diff(a, axis=0)).max(), 0.2)      # drifts, does not jitter

    def test_spell_weights_make_short_spells_count_more(self):
        gt = np.array([0] * 50 + [1] * 400 + [2] * 1000)
        w = m7.spell_weights(gt)
        self.assertEqual(w[0], 4.0)
        self.assertEqual(w[100], 0.5)
        self.assertEqual(w[-1], 0.25)

    def test_controlled_merges_uncontrolled_touches_only(self):
        H, A, D = 0, 1, 2
        gt = np.array([H] * 100 + [A] * 10 + [H] * 5 + [A] * 100 + [D] * 50 + [A] * 10 + [H] * 100)
        out = m7.controlled(gt, 50)
        self.assertTrue((out[:115] == H).all())       # A's 10-frame touch stays H's possession
        self.assertTrue((out[115:215] == A).all())    # a controlled A run is a real change
        np.testing.assert_array_equal(out[215:], gt[215:])   # short run after a restart is kept
        np.testing.assert_array_equal(m7.controlled(gt, 1), gt)


class ModelTests(unittest.TestCase):

    def test_logistic_recovers_a_separable_rule(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, 3))
        y = (X[:, 0] - 2 * X[:, 1] + 0.5 + rng.logistic(size=4000) * 0.1 > 0) * 1.0
        model = m7.fit_logistic(X, y)
        acc = ((m7.logit(model, X) > 0) == (y > 0)).mean()
        self.assertGreater(acc, 0.95)
        self.assertLess(abs(model["beta"][3]), 0.3 * abs(model["beta"][1]))   # the unused feature stays small

    def test_weights_round_trip_through_json(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 2))
        model = m7.fit_logistic(X, (X[:, 0] > 0) * 1.0)
        with tempfile.TemporaryDirectory() as folder:
            path = m7.save_weights(model, model, ["a", "b"], ["a", "b"], os.path.join(folder, "w.json"))
            back = m7.load_weights(path)
        self.assertTrue(np.array_equal(back["team"]["beta"], model["beta"]))       # exact, not close
        self.assertTrue(np.array_equal(back["dead"]["sd"], model["sd"]))
        self.assertEqual(back["team"]["features"], ["a", "b"])


class FeatureTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ball, cls.players, cls.team, cls.half = toy_match()
        cls.ev, _ = m2.build_evidence(cls.ball, cls.players, cls.team, PITCH)

    def test_feature_matrices_are_finite_and_named(self):
        XT, XD, names_team, names_dead = m7.features(self.ev, self.players, self.team, self.half)
        self.assertEqual(XT.shape, (len(self.ball), len(names_team)))
        self.assertEqual(XD.shape, (len(self.ball), len(names_dead)))
        self.assertTrue(np.isfinite(XT).all() and np.isfinite(XD).all())
        self.assertEqual(len(set(names_team)), len(names_team))
        self.assertEqual(len(set(names_dead)), len(names_dead))

    def test_committed_weights_match_the_feature_list(self):
        _, _, names_team, names_dead = m7.features(self.ev, self.players, self.team, self.half)
        weights = m7.load_weights()
        self.assertEqual(weights["team"]["features"], names_team)
        self.assertEqual(weights["dead"]["features"], names_dead)
        self.assertEqual(len(weights["team"]["beta"]), len(names_team))            # no intercept
        self.assertEqual(len(weights["dead"]["beta"]), len(names_dead) + 1)

    def test_probabilities_are_contract_valid_and_make_sense(self):
        p_dead, p_home = m7.probabilities(self.ev, self.players, self.team, self.half)
        ev = dict(contract.empty(len(self.ball)), p_dead=p_dead, p_home=p_home)
        contract.check(ev, len(self.ball))
        self.assertGreater(np.median(p_home[100:800]), 0.6)          # ball stays on the home line
        self.assertGreater(p_dead[950:1050].mean(), p_dead[100:800].mean())   # long gap looks more dead

    def test_swapping_the_teams_flips_p_home(self):
        _, p_home = m7.probabilities(self.ev, self.players, self.team, self.half)
        swapped = (1 - self.team).astype(np.int8)
        ev, _ = m2.build_evidence(self.ball, self.players, swapped, PITCH)
        _, p_swapped = m7.probabilities(ev, self.players, swapped, self.half)
        self.assertTrue(np.allclose(p_home + p_swapped, 1.0, atol=1e-6))

    def test_stale_weights_are_refused(self):
        weights = m7.load_weights()
        weights["team"]["features"] = weights["team"]["features"][::-1]
        with self.assertRaises(ValueError):
            m7.probabilities(self.ev, self.players, self.team, self.half, weights)


if __name__ == "__main__":
    unittest.main()
