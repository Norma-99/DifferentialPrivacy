from differential_privacy.dataset import Dataset
from .network_component import NetworkComponent


class Server(NetworkComponent):
    def __init__(self, validation_dataset: Dataset):
        NetworkComponent.__init__(self)
        self.validation_dataset = validation_dataset
