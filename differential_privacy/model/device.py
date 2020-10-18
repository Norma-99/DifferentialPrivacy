import logging
from differential_privacy.datasets import Dataset
from .network_component import NetworkComponent


logger = logging.getLogger(__name__)


class Device(NetworkComponent):
    def __init__(self, dataset: Dataset):
        NetworkComponent.__init__(self)
        self.dataset = dataset
        self.fog_node_address = None

    def set_fog_node(self, fog_node_address: int):
        self.fog_node_address = fog_node_address

    def send_data(self):
        self.send({'dataset': self.dataset}, self.fog_node_address)
