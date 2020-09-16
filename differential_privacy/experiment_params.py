import importlib
from differential_privacy.dataset import Dataset
from differential_privacy.model_manager import ModelManager
from differential_privacy.utils import build_model


class ExperimentParams:
    def __init__(self, config:dict):
        self.splits = list(Dataset.load_splits(config['splits_path'], config['devices']))
        self.validation = Dataset.load_validation(config['val_path'])
        self.iterations = config['iterations']
        self.model = build_model(config)
        encryptor_name = config['encryptor_name']
        lib = importlib.import_module(f'differential_privacy.encryptors.{encryptor_name}')
        self.encryptor = lib.ExperimentEncryptor()
        self.fog_nodes = config['fog_nodes']
        self.devices = config['devices']
        self.model_manager = ModelManager(config)