import unittest
import numpy as np
from differential_privacy.gradient import Gradient


ONES_DATA = [np.ones((3, 3)), np.ones((3, 1))]


class FolderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.gradients = [FolderTestCase._create_gradient(i) for i in range(3)]

    @staticmethod
    def _create_gradient(value):
        return Gradient(ONES_DATA) * value


if __name__ == '__main__':
    unittest.main()
