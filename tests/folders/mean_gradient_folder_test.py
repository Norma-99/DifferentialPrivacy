import unittest
from .folder_test import FolderTestCase
from differential_privacy.factories.gradient_factory import GradientFactory


class MeanGradientFolderTest(FolderTestCase):
    def test_folding(self):
        folder = GradientFactory.from_name('mean')
        self.assertEqual(folder.fold(self.gradients), MeanGradientFolderTest._create_gradient(1))


if __name__ == '__main__':
    unittest.main()
