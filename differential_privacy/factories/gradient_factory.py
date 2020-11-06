import typing
from differential_privacy.dataset import Dataset
from differential_privacy.folders import GradientFolder, PassGradientFolder, MeanGradientFolder, PonderatedGradientFolder, ThresholdGradientFolder


class GradientFactory:
    @staticmethod
    def from_name(name: str, **kwargs):
        if name == 'mean':
            return MeanGradientFolder()
        elif name == 'ponderated':
            return PonderatedGradientFolder(**kwargs)
        elif name == 'threshold':
            return ThresholdGradientFolder()
        elif name == 'pass': 
            return PassGradientFolder()
