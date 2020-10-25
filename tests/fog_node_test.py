import unittest
import numpy as np
from differential_privacy.model import FogNode


class StubGradientFolder:
    def __init__(self):
        self.gradients = None

    def fold(gradients):
        self.gradients = gradients


class StubModel:
    def fit(dataset): 
        return [np.ones(3, 3), np.ones(3, 1)]


class FogNodeTestCase(unittest.TestCase):
    def setUp(self):
        self.gradient_folder = StubGradientFolder()
        self.fog_node = FogNode(3, self.gradient_folder)

    def test_model_request_on_dataset_sending(self):
        pass

    def test_gradient_fold_on_iteration_end(self):
        pass


if __name__ == '__main__':
    unittest.main()
