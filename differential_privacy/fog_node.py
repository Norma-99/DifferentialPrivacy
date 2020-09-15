from differential_privacy.device import Device
from differential_privacy.encryptor import Encryptor
from differential_privacy.gradient_operations import gradient_calc
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model

class FogNode:
    available_id = 0

    def __init__(self, datasets:list, encryptor:Encryptor, epochs):
        self.devices = [Device(dataset) for dataset in datasets]
        self.encryptor = encryptor
        self.epochs = epochs
        self.id = FogNode.available_id
        FogNode.available_id += 1

    def get_enc_gradient(self, model):
        gradients = [self._train(clone_model(model), device) for device in devices]
        return self.encryptor.encript(gradients)

    def _train(self, model:Model, device):
        initial_weights = model.get_weights()
        model.fit(*device.dataset.get(), epochs=self.epochs)
        return gradient_calc(initial_weights, model.get_weights())