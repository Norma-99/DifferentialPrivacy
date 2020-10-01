from .dataset import Dataset

class DeviceDataset(Dataset):
    def __init__(self, x, y):
        self.Dataset.__init__(self, x, y)
        self._device_id = None

    def get_device_id(self):
        if self._device_id is None:
            raise ValueError('Getting device id from dataset before being set')
        return self._device_id
    
    def set_device_id(self, device_id):
        if self._device_id is not None:
            raise ValueError('Setting an already existing id to the dataset')
        self._device_id = device_id