import logging
from .network_component import NetworkComponent
from .neural_network import NeuralNetwork
from differential_privacy.dataset import Dataset


logger = logging.getLogger(__name__)


class FogNode(NetworkComponent):
    def __init__(self, device_count: int, gradient_folder):
        NetworkComponent.__init__(self)
        self._device_count: int = device_count
        self._gradient_folder = gradient_folder
        self._server_address: int = 0
        self._gradients: List[Gradient] = []
        self._generalization_dataset = {}
        self._current_dataset: Dataset = Dataset(None, None)
        self._current_device: int = None

    def on_data_receive(self, data: dict):
        logger.debug("Received %s", str(data))
        if 'dataset' in data:
            self._process_dataset(data)
        elif 'neural_network' in data:
            self._train_network(data['neural_network'])
            if self._has_all_gradients():
                self.on_iteration_end(data['neural_network'])

    def _save_generalisation_fragment(self):
        self._generalization_dataset[self._current_device] = self._current_dataset.get_generalisation_fragment(self._device_count)
        self._current_device = self._current_device + 1

    def _process_dataset(self, data: dict):
        self._current_dataset = data['dataset']
        self._current_device = data['origin']
        logger.info('Processed dataset from %d', self.current_device)
        self._save_generalisation_fragment()
        self.send({'type': 'neural_network_request'}, self.server_address)

    def _train_network(self, neural_network: NeuralNetwork):
        self._gradients.append(neural_network.fit(self._current_dataset))

    def set_server(self, server_address: int):
        self.server_address = server_address

    def on_iteration_end(self, neural_network: NeuralNetwork):
        gradient = self.gradient_folder.fold(neural_network, self.gradients)
        self.gradients.clear()
        self.send(gradient, self.server_address)

    def _has_all_gradients(self):
        return len(self._current_device) == self.device_count
