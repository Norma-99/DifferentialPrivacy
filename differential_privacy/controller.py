import logging
from typing import List
from differential_privacy.datasets import Dataset
from differential_privacy.model.server import Server
from differential_privacy.factories.fog_node_factory import FogNodeFactory
from differential_privacy.model.fog_node import FogNode


logger = logging.getLogger(__name__)


class Controller:
    def __init__(self, config):
        server = self._create_server(config['server_config'])
        logger.info('Server created at %d', server.get_address())
        fog_nodes = self._create_fog_nodes(config, server)

    @staticmethod
    def _create_server(server_config: dict) -> Server:
        validation_dataset = Dataset.from_file(server_config['dataset_path'])
        return Server(validation_dataset)

    @staticmethod
    def _create_fog_nodes(config: dict, server: Server) -> List[FogNode]:
        fog_node_factory = FogNodeFactory(config['fog_node_config'])
        fog_nodes = [fog_node_factory.create_fog_node() for _ in range(config['fog_node_count'])]
        logger.info('All fog nodes created')
        for fog_node in fog_nodes:
            fog_node.set_server(server.get_address())

    def run(self):
        pass
