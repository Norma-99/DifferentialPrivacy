import unittest
import numpy as np
from differential_privacy.model import NetworkComponent, Server, Network
from differential_privacy.gradient import Gradient


GRADIENT_DATA = [np.ones((3, 3)), np.ones((3, 1)), np.ones((3, 3)), np.ones((3, 1))]


class FogNodeStub(NetworkComponent):
    def __init__(self, server_address):
        NetworkComponent.__init__(self)
        self.server_address = server_address
        self.network_received = None

    def send_request(self):
        self.send({}, self.server_address)

    def send_weights(self):
        self.send({'gradient': Gradient(GRADIENT_DATA)}, self.server_address)

    def on_data_receive(self, data: dict):
        if 'neural_network' in data:
            self.network_received = data['neural_network']


class NeuralNetworkStub:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = Gradient(GRADIENT_DATA)
        else:
            self.weights = weights

    def evaluate(self):
        return {'acc': 0.5}

    def clone(self):
        return NeuralNetworkStub(self.weights)

    def apply_gradient(self, gradient):
        self.weights += gradient


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.neural_network = NeuralNetworkStub()
        self.server = Server(self.neural_network)
        self.fog_node_stub = FogNodeStub(self.server.get_address())
        self.network = Network()
        self.network.add_component(self.server)
        self.network.add_component(self.fog_node_stub)

    def test_network_sending(self):
        self.fog_node_stub.send_request()
        self.assertIsNotNone(self.fog_node_stub.network_received)

    def test_iteration_end(self):
        self.fog_node_stub.send_request()
        initial_weights = self.fog_node_stub.network_received.weights
        self.fog_node_stub.send_weights()
        self.fog_node_stub.send_request()
        final_weights = self.fog_node_stub.network_received.weights
        self.assertEqual(final_weights, initial_weights + Gradient(GRADIENT_DATA))


if __name__ == '__main__':
    unittest.main()