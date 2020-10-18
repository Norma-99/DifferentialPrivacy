import logging
from .network_component import NetworkComponent
from differential_privacy.datasets import Dataset


logger = logging.getLogger(__name__)


class FogNode(NetworkComponent):
    def __init__(self, device_count: int, gradient_folder):
        NetworkComponent.__init__(self)
        self.device_count = device_count
        self.gradient_folder = gradient_folder
        self.server_address = None
        self.gradients = []
        self.dataset_fragments = {}
        self.current_dataset: Dataset = Dataset(None, None)
        self.current_device = None

    def on_data_receive(self, data: dict):
        logger.debug("Received %s", str(data))
        if 'dataset' in data:
            # TODO: Save fragment
            self._process_dataset(data)
        elif 'neural_network' in data:
            self._train_network(data['neural_network'])
            if self._has_all_gradients():
                self.on_iteration_end()

    def _process_dataset(self, data: dict):
        self.current_dataset = data['dataset']
        self.current_device = data['origin']
        self._save_dataset_fragment()
        logger.info('Processed dataset from %d', self.current_device)
        self.send({'type': 'neural_network_request'}, self.server_address)

    def _save_dataset_fragment(self):
        # TODO: Move to dataset class
        x, y = self.current_dataset.get()
        fragment_size = len(x) // self.device_count
        self.dataset_fragments[self.current_device] = Dataset(x[:fragment_size], y[:fragment_size])

    def _train_network(self, neural_network):
        pass

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
