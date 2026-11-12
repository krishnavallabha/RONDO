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
