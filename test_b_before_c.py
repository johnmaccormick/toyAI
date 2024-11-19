import unittest
import basic_transformer as bt
import corpus
import b_before_c as bc


class Test_B_before_C(unittest.TestCase):

    def test_manual_weights(self):
        seed = 12121
        num_in_strs = 1000
        min_len = 10
        max_len = min_len
        model, corp = bc.manual_weights(seed, num_in_strs, min_len, max_len)
        prob_y = bc.eval_string(model, corp, 'v v v u u u b x x x c x x <EOS>')
        self.assertAlmostEqual(prob_y, 0.6969711, places=5)
        prob_y = bc.eval_string(model, corp, 'v v v u u u c x x x b x x <EOS>')
        self.assertAlmostEqual(prob_y, 0.2764442, places=5)


if __name__ == '__main__':
    unittest.main()
