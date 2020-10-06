from differential_privacy.datasets import Dataset

class Server:
    def __init__(self, validation_dataset:Dataset):
        self.validation_dataset = validation_dataset

    def request_model(self):
        pass

    def on_gradient_receive(self):
        pass