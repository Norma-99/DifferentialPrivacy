from differential_privacy.folders import GradientFolder, PassGradientFolder, MeanGradientFolder, PonderatedGradientFolder, ThresholdGradientFolder


class GradientFactory:
    @staticmethod
    def from_name(name: str, **kwargs) -> GradientFolder:
        if name == 'mean':
            return MeanGradientFolder()
        elif name == 'ponderated':
            return PonderatedGradientFolder(**kwargs)
        elif name == 'threshold':
            return ThresholdGradientFolder()
        elif name == 'pass': 
            return PassGradientFolder()
        raise ValueError('Folder not found')
