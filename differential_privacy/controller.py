import importlib
from differential_privacy.dataset import Dataset
from differential_privacy.utils import build_model


class Controller:
    pass


class ExperimentParams:
    def __init__(self, config:dict):
        self.splits = Dataset.load_splits(config['splits_path'])
        self.validation = Dataset.load_validation(config['val_path'])
        self.epochs = config['epochs']
        self.iterations = config['iterations']
        self.model = build_model(config)
        encryptor_name = config['encryptor_name']
        lib = importlib.import_module(f'differential_privacy.encryptors.{encryptor_name}')
        self.encryptor = lib.ExperimentEncryptor()


