import math
import numpy as np
import unittest

import distances
import models


class TestModels(unittest.TestCase):

    def test_cca(self):
        k = 2
        T = 1
        cca = models.CCA(k, T)

        state = np.random.randint(low=0, high=k, size=(20, 20))
        next_state = cca.update(state)
        self.assertTrue(np.all(0 <= state) and np.all(state < 2))
        self.assertEqual(state.shape, next_state.shape)
        self.assertEqual('foo'.upper(), 'FOO')

        state = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        expected = np.ones_like(state)
        self.assertTrue(np.array_equal(cca.update(state), expected))


class TestDistances(unittest.TestCase):

    def test_l2(self):
        p = {1 : 4, 2 : 5, 3 : 7}
        q = {1 : 5, 2 : 6, 4 : 9}
        expected = math.sqrt((5 / 16 - 6 / 20)**2 + (7 / 16) **2 + (9 / 20) ** 2)
        self.assertEquals(distances.l2(p, q), expected)
        self.assertEquals(distances.l2(q, p), expected)


if __name__ == '__main__':
    unittest.main()
