#This will import ADULT dataset and create an MLP to train the Dataset
import argparse
import json

import pickle
import tensorflow as tf
import numpy as np
import random

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = {}
    with open(args.config) as conf_file:
        config = json.load(conf_file)

    # Load data
    print("Loading data")
    x_train, y_train = load_data(config['train_dataset'])
    x_test, y_test = load_data(config['val_dataset'])

    # Create network
    model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(74,)),
    tf.keras.layers.Dense(config['first_layer_units'], activation='relu', input_shape=(87,)),
    tf.keras.layers.Dense(config['second_layer_units'], activation='relu'), # 2/3 input + output
    tf.keras.layers.Dense(config['third_layer_units'], activation='relu'),
    tf.keras.layers.Dense(config['sigmoid_layer_units'], activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['accuracy'])
    
    # Train network
    print("training network")
    model.fit(x_train, y_train, epochs=config['epochs'], validation_data=(x_test, y_test))

