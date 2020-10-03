import logging
from differential_privacy.datasets import DeviceDataset
from differential_privacy.model.fog_node import FogNode

logger = logging.getLogger(__name__)


class Device:
    available_id = 0

    def __init__(self, dataset:DeviceDataset, fog_node:FogNode=None):
        self.id = Device.available_id
        Device.available_id += 1
        self.dataset = dataset
        self.dataset.set_device_id(self.id)
        self.fog_node = fog_node
        logger.info('Created device %d', self.id)

    def set_fog_node(self, fog_node:FogNode):
        if self.fog_node is not None:
            logger.warning('Resetting fog node on device %d', self.id)
        self.fog_node = fog_node

    def send_data(self):
        logger.info('Sent from %d to %d', self.id, self.fog_node.id)
        self.fog_node.on_data_receive(self.dataset)
