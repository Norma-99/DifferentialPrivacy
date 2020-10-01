import logging
from differential_privacy.datasets import DeviceDataset

class Device:
    available_id = 0
    logger = logging.getLogger(__name__)

    def __init__(self, dataset:DeviceDataset):
        self.id = Device.available_id
        Device.available_id += 1
        self.dataset = dataset
        self.dataset.set_device_id(self.id)
        Device.logger.info(f'Created device {self.id}')
