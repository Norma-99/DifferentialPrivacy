import typing
from typing import Dict
import tensorflow as tf
from differential_privacy.dataset import Dataset
from differential_privacy.gradient import Gradient


class NeuralNetwork:
    def __init__(self, tf_model, epochs, validation_dataset):
        self.tf_model: tf.keras.model = tf_model
        self.epochs: int = epochs
        self.validation_dataset: Dataset = validation_dataset

    def fit(self, data:Dataset) -> Gradient:
        initial_weights = self.tf_model.get_weights()
        self.tf_model.fit(*data.get(), epochs=self.epochs)  # TODO: Add callback
        final_weights = self.tf_model.get_weights()
        return Gradient.from_delta(initial_weights, final_weights)

    def clone(self):
        # Returns a NeuralNetwork
        pass

    def evaluate(self) -> Dict[str, float]:
        pass

    def apply_gradient(self, gradient: Gradient):
        pass
