import unittest
from differential_privacy.model.device import Device
from differential_privacy.model.fog_node import FogNode
from differential_privacy.datasets import DeviceDataset


class DeviceTestCase(unittest.TestCase):
    def test_constructor(self):
        dataset = DeviceDataset(None, None)
        device = Device(dataset)
        with self.assertRaises(AttributeError):
            device.send_data()

    def test_fog_node_setter(self):
        dataset = DeviceDataset(None, None)
        device = Device(dataset)
        fog_node = FogNode()
        device.set_fog_node(fog_node)
        self.assertEqual(device.fog_node, fog_node)


if __name__ == '__main__':
    unittest.main()
