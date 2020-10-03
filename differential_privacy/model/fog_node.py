from differential_privacy.datasets import DeviceDataset


class FogNode:
    def __init__(self):
        self.id = None

    def on_data_receive(self, data:DeviceDataset):
        pass