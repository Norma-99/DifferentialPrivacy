import numpy as np
import unittest
from differential_privacy.gradient import Gradient

ZEROS = [np.zeros((3, 3)), np.zeros((3, 1))]
ONES = [np.ones((3, 3)), np.ones((3, 1))]


class MyTestCase(unittest.TestCase):
    def test_gradient_creation(self):
        gradient1 = Gradient(ZEROS)
        gradient2 = Gradient.from_delta(ONES, ONES)
        self.assertEqual(gradient1, gradient2)

    def test_gradient_multiplication(self):
        gradient1 = Gradient(ONES)
        self.assertEqual(gradient1 * 0, Gradient(ZEROS))
        self.assertEqual(gradient1 * 2, gradient1 + gradient1)


if __name__ == '__main__':
    unittest.main()
