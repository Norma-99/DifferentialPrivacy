import unittest
from .folder_test import FolderTestCase
from differential_privacy.factories.gradient_factory import GradientFactory
from differential_privacy.dataset import Dataset


# 0.5 / (0.5 + 0.5 + 0.5) * Gradient(0) + 0.5 / (0.5 + 0.5 + 0.5) * Gradient(1) + 0.5 / (0.5 + 0.5 + 0.5) * Gradient(2)
EXPECTED = 1


class NeuralNetworkStub:
    def __init__(self):
        self.call = 0
        self.dataset = None

    def clone(self):
        return self

    def apply_gradient(self, gradient):
        pass

    def evaluate(self, dataset):
        self.dataset = dataset
        if self.call == 0:
            return {'acc': 0.5}
        elif self.call == 1:
            return {'acc': 0.5}
        elif self.call == 2:
            return {'acc': 0.5}


class PonderatedGradientFolderTest(FolderTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_folding(self):
        neural_network = NeuralNetworkStub()
        dataset = Dataset(None, None)
        folder = GradientFactory.from_name('ponderated', neural_network=neural_network, dataset=dataset)
        self.assertEqual(folder.fold(self.gradients), FolderTestCase._create_gradient(EXPECTED))
        self.assertIs(neural_network.dataset, dataset)


if __name__ == '__main__':
    unittest.main()