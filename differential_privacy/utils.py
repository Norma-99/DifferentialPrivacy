import json
from tensorflow.keras.layers import GaussianNoise, Dense, InputLayer
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.metrics import AUC, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from differential_privacy.dataset import Dataset

def read_config(path:str):
    with open(path) as conf_file:
        return json.load(conf_file)


def build_model(config:dict):
    model = Sequential([InputLayer(input_shape=(87,))]) #(87,) or (74,)
    if config['has_noise']:
        model.add(GaussianNoise(config['noise_variance']))
    for i in range(config['hidden_count']):
        model.add(Dense(config[f'layer{i}_units'], activation='relu'))
    model.add(Dense(config['sigmoid_layer_units'], activation='sigmoid'))
    model.compile(optimizer=config['optimizer'],
        loss=config['loss'],
        metrics=['accuracy', AUC(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives()])
    model.summary()
    return model