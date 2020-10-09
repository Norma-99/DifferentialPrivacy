import logging
from differential_privacy.datasets import Dataset
from differential_privacy.model.fog_node import FogNode


logger = logging.getLogger(__name__)


class Device:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.fog_node_address = None

    def set_fog_node(self, fog_node: FogNode):
        if self.fog_node is not None:
            logger.warning('Resetting fog node on device %d', self.id)
        self.fog_node = fog_node

    def send_data(self):
        logger.info('Sent from %d to %d', self.id, self.fog_node.id)
        self.fog_node.on_data_receive(self.dataset)
