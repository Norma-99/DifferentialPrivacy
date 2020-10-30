import unittest
import numpy as np
from differential_privacy.dataset import Dataset
from differential_privacy.model import FogNode, Device, Server, Network, NeuralNetwork
from differential_privacy.gradient import Gradient


DATASET_X = np.ones((3, 3))
DATASET_Y = np.ones((3, 1))
TRAINING_GRADIENT = Gradient(None)


class DeviceStub(Device):
    def __init__(self):
        Device.__init__(self, None)

    def send_dataset(self, to: int):
        self.send({'dataset': Dataset(DATASET_X, DATASET_Y)}, to)


class RequestCheckerServerStub(Server):
    def __init__(self):
        Server.__init__(self, None)
        self.model_request_received = False

    def on_data_receive(self, data: dict):
        if 'gradient' not in data:
            self.model_request_received = True


class NeuralNetworkStub(NeuralNetwork):
    def __init__(self):
        NeuralNetwork.__init__(self, None, None, None)

    def fit(self, dataset):
        return TRAINING_GRADIENT


class IterationCheckerServerStub(Server):
    def __init__(self):
        Server.__init__(self, None)
        self.received_gradient = None

    def on_data_receive(self, data: dict):
        if 'gradient' in data:
            self.received_gradient = data['gradient']
        else:
            self.send({'neural_network': NeuralNetworkStub()}, to=data['origin'])

    def is_gradient_correct(self):
        return self.received_gradient is not None and self.received_gradient is TRAINING_GRADIENT


class FogNodeTestCase(unittest.TestCase):
    def setUp(self):
        self.fog_node_count = 2
        self.fog_node = FogNode(self.fog_node_count, gradient_folder='pass')
        self.network = Network()
        self.network.add_component(self.fog_node)
        self.device_stubs = self._create_device_stubs()

    def _create_device_stubs(self):
        stubs = []
        for _ in range(self.fog_node_count):
            stub = DeviceStub()
            self.network.add_component(stub)
            stubs.append(stub)
        return stubs

    def _add_server(self, server_stub: Server):
        self.fog_node.set_server(server_stub.get_address())
        self.network.add_component(server_stub)
        return server_stub

    def test_model_request_on_dataset_sending(self):
        server_stub = self._add_server(RequestCheckerServerStub())
        self.device_stubs[0].send_dataset(self.fog_node.get_address())
        self.assertTrue(server_stub.model_request_received)

    def test_gradient_fold_on_iteration_end(self):
        server_stub = self._add_server(IterationCheckerServerStub())
        for device_stub in self.device_stubs:
            device_stub.send_dataset(self.fog_node.get_address())
        self.assertTrue(server_stub.is_gradient_correct())


if __name__ == '__main__':
    unittest.main()
