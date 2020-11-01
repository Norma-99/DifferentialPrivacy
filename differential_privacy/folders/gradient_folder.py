from typing import List
from differential_privacy.gradient import Gradient
from differential_privacy.dataset import Dataset


class GradientFolder:
    def __init__(self):
        self.generalization_dataset = GeneralizationDataset()

    def fold(self, neural_network, gradients: List[Gradient]) -> Gradient:
        raise NotImplementedError()

    