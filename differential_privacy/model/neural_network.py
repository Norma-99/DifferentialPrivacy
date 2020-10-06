from differential_privacy.datasets.device_dataset import DeviceDataset
from differential_privacy.gradient_operations import gradient_calc

class NeuralNetwork:
    def __init__(self, tf_model, epochs):
        self.tf_model = tf_model
        self.epochs = epochs

    def fit(self, data:DeviceDataset) -> list:
        initial_weights = self.tf_model.get_weights()
        self.tf_model.fit(*data.get(), epochs=self.epochs)  # TODO: Add callback
        final_weights = self.tf_model.get_weights()
        return gradient_calc(initial_weights, final_weights)

    def clone(self):
        pass

    def evaluate(self):
        pass

