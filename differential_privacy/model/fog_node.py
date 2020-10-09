from .network_component import NetworkComponent


class FogNode(NetworkComponent):
    def __init__(self, device_count: int, gradient_folder):
        NetworkComponent.__init__(self)
        self.server_address = None
        self.gradients = []
        self.device_count = device_count
        self.gradient_folder = gradient_folder

    def on_data_receive(self, data):
        # TODO: Rework
        self.gradient_folder.add_subdataset(data)
        neural_network = self.server.request_model()
        gradient = neural_network.fit(data)
        self.gradients.append(gradient)
        if self._has_all_gradients():
            self.on_iteration_end()

    def set_server(self, server_address: int):
        self.server_address = server_address

    def on_iteration_end(self):
        # TODO: Rework
        neural_network = self.server.request_model()
        gradient = self.gradient_folder.fold(neural_network, self.gradients)
        self.gradients.clear()
        self.server.on_gradient_receive(gradient)

    def _has_all_gradients(self):
        return len(self.gradients) == self.device_count
