from differential_privacy.controller import ExperimentParams


class Server:
    def __init__(self, experiment_params:ExperimentParams):
        self.model = experiment_params.model
        

