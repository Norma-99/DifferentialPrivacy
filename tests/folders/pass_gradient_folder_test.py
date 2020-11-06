import unittest
from .folder_test import FolderTestCase
from differential_privacy.factories.gradient_factory import GradientFactory


class PassGradientFolderTest(FolderTestCase):
    def test_folding(self):
        folder = GradientFactory.from_name('pass')
        self.assertEqual(folder.fold(self.gradients), FolderTestCase._create_gradient(0))


if __name__ == '__main__':
    unittest.main()
