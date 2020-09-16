from differential_privacy.server import Server
from differential_privacy.experiment_params import ExperimentParams
from differential_privacy.utils import read_config


class Controller:
    def __init__(self, args):
        config = read_config(args.config)
        self.server = Server(ExperimentParams(config))

    def run(self):
        self.server.train()