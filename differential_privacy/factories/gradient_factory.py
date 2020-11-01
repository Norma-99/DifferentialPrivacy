import typing
from differential_privacy.dataset import Dataset
from differential_privacy.folders import GradientFolder, PassGradientFolder, MeanGradientFolder, PonderatedGradientFolder, ThresholdGradientFolder


class GradientFactory:
    @staticmethod
    def from_name(name: str, generalisation_dataset: Dataset):
        if name == 'mean':
            return MeanGradientFolder(generalisation_dataset)
        elif name == 'ponderated':
            return PonderatedGradientFolder(generalisation_dataset)
        elif name == 'threshold':
            return ThresholdGradientFolder(generalisation_dataset)
        elif name == 'pass': 
            return PassGradientFolder(generalisation_dataset)
            