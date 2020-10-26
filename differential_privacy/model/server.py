from differential_privacy.dataset import Dataset
from .network_component import NetworkComponent


class Server(NetworkComponent):
    def __init__(self, validation_dataset: Dataset):
        NetworkComponent.__init__(self)
        self.validation_dataset = validation_dataset

    def on_data_receive(self, data: dict):
        # O bien recibe petición del modelo
        # O bien recibe los pesos del modelo ya entrenado
        pass

    def _process_dataset_request(data: dict):
        pass

    def _process_gradient(data: dict):
        pass