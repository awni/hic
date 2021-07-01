import collections
import math
import numpy as np
import unittest

import measures 
import models


class TestModels(unittest.TestCase):

    def test_eca(self):
        eca = models.ECA(30)
        rule_30 = {
            (0, 0, 0): 0, (0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1,
            (1, 0, 0): 1, (1, 0, 1): 0, (1, 1, 0): 0, (1, 1, 1): 0
        }
        for state, v in rule_30.items():
            self.assertEqual(eca.update(state)[1], v)


    def test_cca(self):
        k = 2
        T = 1
        cca = models.CCA(k, T)

        state = np.random.randint(low=0, high=k, size=(20, 20))
        next_state = cca.update(state)
        self.assertTrue(np.all(0 <= state) and np.all(state < 2))
        self.assertEqual(state.shape, next_state.shape)

        state = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        expected = np.ones_like(state)
        self.assertTrue(np.array_equal(cca.update(state), expected))


class TestMeasures(unittest.TestCase):

    def test_entropy(self):
        H = measures.entropy([2, 2, 2, 2])
        self.assertTrue(math.isclose(H, -math.log(0.25)))

        H = measures.entropy([3, 3, 3, 3])
        self.assertTrue(math.isclose(H, -math.log(0.25)))

        H = measures.entropy([0, 0, 7, 0])
        self.assertTrue(math.isclose(H, 0.0, abs_tol=1e-15))

    def test_conditional_entropy(self):
        condH = measures.conditional_entropy(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        self.assertTrue(math.isclose(condH, -math.log(0.25)))
        condH = measures.conditional_entropy(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertTrue(math.isclose(condH, -math.log(1 / 3.)))
        condH = measures.conditional_entropy(
            [[0, 0, 2], [0, 0, 3], [0, 1, 0], [7, 0, 0]])
        self.assertTrue(math.isclose(condH, 0.0, abs_tol=1e-15))

    def mutual_information(self):
        I = measures.mutual_information(
            [[0, 0, 2], [0, 0, 3], [0, 1, 0], [7, 0, 0]])
        self.assertTrue(math.isclose(I, 0.0, abs_tol=1e-15))

    def test_l2(self):
        p = {1 : 4, 2 : 5, 3 : 7}
        q = {1 : 5, 2 : 6, 4 : 9}
        expected = math.sqrt((5 / 16 - 6 / 20)**2 + (7 / 16) **2 + (9 / 20) ** 2)
        self.assertEqual(measures.l2(p, q), expected)
        self.assertEqual(measures.l2(q, p), expected)


if __name__ == '__main__':
    unittest.main()
