from differential_privacy.dataset import Dataset
from .network_component import NetworkComponent


class Server(NetworkComponent):

    DATASET_REQUEST = 'neural_network_request'

    def __init__(self, validation_dataset: Dataset):
        NetworkComponent.__init__(self)
        self.validation_dataset = validation_dataset

    def on_data_receive(self, data: dict):
        if 'gradient' in data:
            # O bien recibe los pesos del modelo ya entrenado
            pass
        else: 
            # O bien recibe petición del modelo
            pass


    def _process_dataset_request(data: dict):
        pass

    def _process_gradient(data: dict):
        pass