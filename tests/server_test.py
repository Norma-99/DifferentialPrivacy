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

    def send_weights(self, value):
        self.send({'gradient': Gradient(GRADIENT_DATA)*value}, self.server_address)

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
        self.server = Server(self.neural_network, 2)
        self.fog_node_stubs = [FogNodeStub(self.server.get_address()) for _ in range(2)]
        self.network = Network()
        self.network.add_component(self.server)
        for fog_node_stub in self.fog_node_stubs:
            self.network.add_component(fog_node_stub)

    def test_network_sending(self):
        self.fog_node_stubs[0].send_request()
        self.assertIsNotNone(self.fog_node_stubs[0].network_received)

    def test_iteration_end(self):
        # Coger pesos iniciales
        self.fog_node_stubs[0].send_request()
        initial_weights = self.fog_node_stubs[0].network_received.weights

        # Enviar primer gradiente
        self.fog_node_stubs[0].send_weights(0)

        # Comprobar que sigue teniendo los mismos pesos
        self.fog_node_stubs[1].send_request()
        self.assertEqual(self.fog_node_stubs[1].network_received.weights, initial_weights)

        # Enviar segundo gradiente
        self.fog_node_stubs[1].send_weights(2)

        # Comprobar que tiene los pesos finales esperados
        self.fog_node_stubs[0].send_request()
        final_weights = self.fog_node_stubs[0].network_received.weights
        self.assertEqual(final_weights - initial_weights, Gradient(GRADIENT_DATA))


if __name__ == '__main__':
    unittest.main()