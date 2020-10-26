from typing import List
import numpy as np

class Gradient: 
    def __init__(self, layer_deltas):
        self._layer_deltas: List[np.ndarray] = layer_deltas

    @staticmethod
    def from_delta(initial_weights: List[np.ndarray], final_weights: List[np.ndarray]):
        pass

    def apply(weights: List[np.ndarray]) -> List[np.ndarray]:
        pass

    def get() -> List[np.ndarray]:
        pass