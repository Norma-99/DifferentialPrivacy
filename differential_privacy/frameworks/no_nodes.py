from differential_privacy.utils import write_model
from differential_privacy.dataset import Dataset

def run(config:dict):
    write_model(config, has_noise=False)
    dataset = Dataset(
        train_path=config['train_dataset'],
        val_path=config['val_dataset'],
        nodes=config['nodes'])
    validation_cycle(dataset, config, enc_func=lambda x: x)
