from typing import List
from differential_privacy.gradient import Gradient
from differential_privacy.folders.gradient_folder import GradientFolder


class PassGradientFolder(GradientFolder):
    def fold(self, gradients: List[Gradient]) -> Gradient:
        return gradients[0]
