import os
import tempfile
import unittest

import numpy as np

import rondo  # noqa: F401  (puts the challenge repo on sys.path)
import possession_metrics as pm
from games import GAME, GAMES
from rondo import contract, decoder, dummy, submission

N = GAMES[GAME]["n_video_frames"]


class ContractTest(unittest.TestCase):
    def test_empty_passes(self):
        contract.check(contract.empty(N), N)

    def test_missing_key_and_bad_range_rejected(self):
        ev = contract.empty(N)
        del ev["touch_H"]
        ev["dead_ball"][0] = 1.5
        with self.assertRaisesRegex(ValueError, "touch_H.*dead_ball"):
            contract.check(ev, N)


class ViterbiTest(unittest.TestCase):
    def test_sticky_through_one_noisy_frame(self):
        loglik = np.log(np.array([[.9, .05, .05]] * 5 + [[.05, .9, .05]] + [[.9, .05, .05]] * 5))
        path = decoder.viterbi(loglik, decoder.log_transitions(), np.log([1 / 3] * 3))
        self.assertTrue((path == decoder.H).all())


class MinimumDurationTest(unittest.TestCase):
    def test_short_blip_kept_by_plain_hmm_absorbed_with_minimums(self):
        # 4 frames of strong A inside H: worth a switch on their own, not once padded to 10.
        loglik = np.log(np.tile([.8, .1, .1], (200, 1)))
        loglik[100:104] = np.log([.01, .98, .01])
        plain = decoder.decode_half(loglik, dict(decoder.DEFAULTS, min_alive=1, min_dead=1))
        self.assertTrue((plain[100:104] == decoder.A).all())
        self.assertTrue((decoder.decode_half(loglik) == decoder.H).all())

    def test_long_enough_spell_survives(self):
        loglik = np.log(np.tile([.8, .1, .1], (300, 1)))
        loglik[100:200] = np.log([.01, .98, .01])
        codes = decoder.decode_half(loglik)
        self.assertTrue((codes[110:190] == decoder.A).all())

    def test_min_one_is_plain_hmm(self):
        params = dict(decoder.DEFAULTS, min_alive=1, min_dead=1)
        state_of, trans, _ = decoder.expand(params)
        self.assertEqual(len(state_of), 3)
        np.testing.assert_allclose(trans, decoder.log_transitions(params))

    def test_every_decoded_spell_meets_minimum(self):
        rng = np.random.default_rng(1)
        codes = decoder.decode_half(np.log(rng.dirichlet([1, 1, 1], size=3000)))
        edges = np.flatnonzero(np.diff(codes)) + 1
        for a, b in zip(edges[:-1], edges[1:]):  # interior spells only; ends may be cut
            minimum = decoder.DEFAULTS["min_dead" if codes[a] == decoder.D else "min_alive"]
            self.assertGreaterEqual(b - a, minimum)


class RestartTest(unittest.TestCase):
    def test_restart_team_picks_owner_after_stoppage(self):
        n = 120
        ev = contract.empty(n)
        ev["dead_ball"][:60] = 0.95
        ev["dead_ball"][60:] = 0.2
        ev["restart_team"][55:70] = 1.0  # A takes the restart, no control evidence at all
        codes = decoder.decode_half(decoder.emission_loglik(ev))
        self.assertTrue((codes[75:] == decoder.A).all())

    def test_internal_minus_one_rejected(self):
        ev = contract.empty(5)
        ev["restart_team"][:] = -1
        with self.assertRaisesRegex(ValueError, "restart_team"):
            decoder.emission_loglik(ev)


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev, cls.truth = dummy.evidence(N, seed=0)
        cls.labels = decoder.decode(cls.ev, N)

    def test_x_exactly_outside_graded_halves(self):
        graded = np.zeros(N, dtype=bool)
        for start, stop in decoder.graded_blocks():
            graded[start:stop] = True
        self.assertEqual(decoder.graded_blocks(), [(11185, 90652), (110808, 190198)])
        self.assertTrue((self.labels[~graded] == "X").all())
        self.assertFalse((self.labels[graded] == "X").any())

    def test_recovers_synthetic_possessions(self):
        gt = np.where(self.truth < 0, pm.X, self.truth).astype(np.int8)
        score = pm.spell_scores(gt, pm.label_codes(self.labels), pm.valid_runs(gt))["mean_f1"]
        self.assertGreater(score, 0.9)

    def test_csv_valid_and_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
            submission.write(self.labels, a)
            submission.write(decoder.decode(dummy.evidence(N, seed=0)[0], N), b)
            with open(a, "rb") as fa, open(b, "rb") as fb:
                data = fa.read()
                self.assertEqual(data, fb.read())
            self.assertNotIn(b"\r\n", data)


if __name__ == "__main__":
    unittest.main()
