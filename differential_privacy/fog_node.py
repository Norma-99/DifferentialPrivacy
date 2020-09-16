from differential_privacy.device import Device
from differential_privacy.encryptor import Encryptor
from differential_privacy.model_manager import ModelManager
from differential_privacy.gradient_operations import gradient_calc
from tensorflow.keras import Model

class FogNode:
    available_id = 0

    def __init__(self, datasets:list, encryptor:Encryptor, model_manager:ModelManager):
        self.devices = [Device(dataset) for dataset in datasets]
        self.encryptor = encryptor
        self.model_manager = model_manager
        self.id = FogNode.available_id
        FogNode.available_id += 1

    def get_enc_gradient(self, model:Model):
        gradients = [self._train(self.model_manager.clone_model(model), device) for device in self.devices]
        return self.encryptor.encript(gradients)

    def _train(self, model:Model, device):
        initial_weights = model.get_weights()
        self.model_manager.fit(model, self, device)
        return gradient_calc(initial_weights, model.get_weights())