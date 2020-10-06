from differential_privacy.datasets import DeviceDataset
from differential_privacy.model.server import Server


class FogNode:
    def __init__(self):
        self.id = None
        self.server = None
        self.gradients = []
        self.device_count = 6
        self.gradient_folder = None

    def on_data_receive(self, data:DeviceDataset):
        self.gradient_folder.add_subdataset(data)
        neural_network = self.server.request_model()
        gradient = neural_network.fit(data)
        self.gradients.append(gradient)
        if self._has_all_gradients():
            self.on_iteration_end()

    def set_server(self, server:Server):
        self.server = server

    def on_iteration_end(self):
        neural_network = self.server.request_model()
        gradient = self.gradient_folder.fold(neural_network, self.gradients)
        self.gradients.clear()
        self.server.on_gradient_receive(gradient)

    def _has_all_gradients(self):
        return len(self.gradients) == self.device_count
