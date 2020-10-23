import typing
from differential_privacy.gradient import Gradient
from differential_privacy.datasets import Dataset

class GradientFolder:
    def __init__(self):
        self.generalization_dataset = GeneralizationDataset()

    def fold(self, gradients: List[Gradient]) -> Gradient:
        raise NotImplementedError()

    @staticmethod
    def from_name(name: str, generalisation_dataset: Dataset):
        # Returns GradientFolder
        pass