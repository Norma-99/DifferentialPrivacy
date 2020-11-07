import typing
from typing import Dict
import tensorflow as tf
from differential_privacy.dataset import Dataset
from differential_privacy.gradient import Gradient


class NeuralNetwork:
    def __init__(self, tf_model, epochs, validation_dataset):
        self.tf_model: tf.keras.Model = tf_model
        self.epochs: int = epochs
        self.validation_dataset: Dataset = validation_dataset

    def fit(self, data:Dataset) -> Gradient:
        initial_weights = self.tf_model.get_weights()
        self.tf_model.fit(*data.get(), epochs=self.epochs)  # TODO: Add callback
        final_weights = self.tf_model.get_weights()
        return Gradient.from_delta(initial_weights, final_weights)

    def clone(self):
        new_model = tf.keras.models.clone_model(self.tf_model)
        new_model.set_weights(self.tf_model.get_weights())
        return NeuralNetwork(new_model, self.epochs, self.validation_dataset)

    def evaluate(self, dataset: Dataset = None) -> Dict[str, float]:
        if dataset is not None:
            return self.tf_model.evaluate(*dataset.get(), return_dict=True)
        return self.tf_model.evaluate(*self.validation_dataset.get())

    def apply_gradient(self, gradient: Gradient):
        self.tf_model.set_weights(
            (Gradient(self.tf_model.get_weights()) + gradient).get()
        )
