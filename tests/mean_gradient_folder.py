import unittest
import numpy as np
from differential_privacy.gradient import Gradient
from differential_privacy.folders.gradient_folder import GradientFolder


ONES_DATA = [np.ones((3, 3)), np.ones((3, 1))]


class MeanGradientFolderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gradients = [MeanGradientFolderTest._create_gradient(i) for i in range(3)]

    @staticmethod
    def _create_gradient(value):
        return Gradient(ONES_DATA) * value

    def test_folding(self):
        folder = GradientFolder.from_name('mean')
        self.assertEqual(folder.fold(self.gradients), MeanGradientFolderTest._create_gradient(1))


if __name__ == '__main__':
    unittest.main()
