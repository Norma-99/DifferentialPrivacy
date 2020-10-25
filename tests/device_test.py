import unittest
import numpy as np
from differential_privacy.dataset import Dataset
from differential_privacy.model import Network, NetworkComponent, Device


X_SHAPE = (10, 87)
Y_SHAPE = (10, 1)


class DatasetChecker(NetworkComponent):
    def __init__(self):
        NetworkComponent.__init__(self)
        self.dataset = None

    def on_data_receive(self, data: dict):
        self.dataset = data['dataset']


class DeviceTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(np.zeros(X_SHAPE), np.zeros(Y_SHAPE))
        self.network = Network()
        self.checker = DatasetChecker()
        self.device = Device(self.dataset)
        self._set_up_network()

    def _set_up_network(self):
        self.device.set_fog_node(self.checker.get_address())
        self.network.add_component(self.device)
        self.network.add_component(self.checker)

    def test_dataset_sending(self):
        self.checker.send({}, self.device.get_address())
        self.assertEqual(self.dataset, self.checker.dataset)


if __name__ == '__main__':
    unittest.main()
