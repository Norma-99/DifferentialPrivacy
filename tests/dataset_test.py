import unittest
import numpy as np
from differential_privacy.dataset import Dataset


TEST_X = np.ones((10, 3, 3))
TEST_Y = np.ones(10)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset(TEST_X, TEST_Y)

    def test_init(self):
        self.assertEqual(self.dataset.get(), (TEST_X, TEST_Y))

    def test_split(self):
        split = self.dataset.get_split(1, 10)
        self.assertEqual(split, Dataset(TEST_X[1:2], TEST_Y[1:2]))

    def test_generalization_dataset(self):
        sub_dataset = self.dataset.get_generalisation_fragment(10)
        self.assertEqual(sub_dataset, Dataset(TEST_X[0:1], TEST_Y[0:1]))


if __name__ == '__main__':
    unittest.main()
