from differential_privacy.dataset import Dataset

class Device:
    available_id = 0

    def __init__(self, dataset:Dataset):
        self.dataset = dataset
        self.id = Device.available_id
        Device.available_id += 1
