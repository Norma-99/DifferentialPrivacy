import unittest
import logging
from differential_privacy.model.device import Device
from differential_privacy.datasets import DeviceDataset

class MyTestCase(unittest.TestCase):
    def test_contstructor(self):
        logging.basicConfig(level=logging.DEBUG)
        dataset = DeviceDataset(None, None)
        Device(dataset)


if __name__ == '__main__':
    unittest.main()
