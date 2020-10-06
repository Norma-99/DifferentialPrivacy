import unittest
from differential_privacy.model.fog_node import FogNode
from differential_privacy.model.server import Server

class FogNodeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.fog_node = FogNode()

    def test_server_setter(self):
        server = Server(None)
        self.fog_node.set_server(server)
        self.assertEqual(self.fog_node.server, server)



if __name__ == '__main__':
    unittest.main()
