from . import NetworkComponent, NeuralNetwork
from differential_privacy.factories.gradient_factory import GradientFactory


class Server(NetworkComponent):
    def __init__(self, neural_network: NeuralNetwork, fog_node_count: int):
        NetworkComponent.__init__(self)
        self.neural_network = neural_network
        self.gradients = []
        self.fog_node_count = fog_node_count

    def on_data_receive(self, data: dict):
        if 'gradient' in data:
            self._process_gradient(data)
        else:
            self.send({'neural_network': self.neural_network.clone()}, data['origin'])

    def _process_gradient(self, data: dict):
        self.gradients.append(data['gradient'])
        if len(self.gradients) == self.fog_node_count:
            gradient = GradientFactory.from_name('mean').fold(self.gradients)
            self.gradients.clear()
            self.neural_network.apply_gradient(gradient)
            self.neural_network.save_trace(self.get_address())
