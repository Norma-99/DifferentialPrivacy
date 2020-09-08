#This will import ADULT dataset and create an MLP to train the Dataset
#This is MLP with the geometric mean delta (later try the threshold)
import argparse
import json
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras.metrics import AUC, FalseNegatives, FalsePositives, TrueNegatives, TruePositives

MODEL_SAVE_PATH = 'temp_mlp.h5'
DATASET_PATH = 'datasets/extended/test/test_dataset.pickle'

class DatasetSplitter:

    def __init__(self, path, nodes):
        self.x, self.y = load_data(path)
        self.nodes = nodes

    def generate_split(self, node_id):
        """Returns the splits acording to the specified arguments."""
        split_size = math.floor(len(self.x) / self.nodes)
        
        start = node_id * split_size
        end = start + split_size

        return self.x[start:end], self.y[start:end]


class ReportCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.results = list()
        self.node = -1 #-1 validation node

    def on_epoch_end(self, epoch, logs=None):
        self.results.append({
            'node': self.node,
            'epoch': epoch,
            **logs
        })

    def save(self, path):
        df = pd.DataFrame(self.results)
        df.to_csv(path)


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_gradient(weights, gradient):
    iteration_deltas = []
    for inital_layer_weights, final_layer_weights in zip(weights, gradient):
        iteration_deltas.append(final_layer_weights - inital_layer_weights)
    return iteration_deltas


def get_ratio_thresh_delta(delta):
    ratio_thresh_deltas = []
    nodes = range(0, len(deltas))
    layers = range(0, len(deltas[0]))
    for layer in layers:
        layer_weights = list()
        layer_max_delta = deltas[0][layer]
        for node in nodes:
            layer_weights.append(deltas[node][layer])
            layer_max_delta = np.maximum(layer_max_delta, deltas[node][layer])
        #Define a random number between median and a 50% between median and maximun
        layer_median_delta = np.median(layer_weights, axis=0)
        layer_difference_delta = (layer_max_delta - layer_median_delta)/2
        ratio_thresh_deltas.append(random.uniform(layer_median_delta, layer_median_delta + layer_difference_delta))
    return ratio_thresh_deltas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = {}
    with open(args.config) as conf_file:
        config = json.load(conf_file)

    callback = ReportCallback()

    # Create network
    model = tf.keras.Sequential([
        tf.keras.layers.GaussianNoise(0.01, input_shape=(87,)),  #(87,) or (74,)
        tf.keras.layers.Dense(config['first_layer_units'], activation='relu'),
        tf.keras.layers.Dense(config['second_layer_units'], activation='relu'),
        tf.keras.layers.Dense(config['third_layer_units'], activation='relu'),
        tf.keras.layers.Dense(config['sigmoid_layer_units'], activation='sigmoid')
    ])
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=['accuracy', AUC(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives()])
    model.summary()
    model.save(MODEL_SAVE_PATH)
    del model

    dataset_splitter = DatasetSplitter(path=DATASET_PATH, nodes=config['nodes'])
    test_data = load_data(config['val_dataset'])

    for iteration in range(int(config['iterations'])):
        deltas = []
        for i in range(config['nodes']):
            
            # Copy net
            print("Copying net")
            iteration_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            initial_weights = iteration_model.get_weights()
            
            # Load data
            print("Loading data")
            x_train, y_train = dataset_splitter.generate_split(i)

            # Train network
            print("training network")
            callback.node = i
            iteration_model.fit(x_train, y_train, epochs=config['epochs'], callbacks=[callback])

            # Save delta
            final_weights = iteration_model.get_weights()
            iteration_deltas = save_gradient(initial_weights, final_weights)
            deltas.append(iteration_deltas)

        # Calculate gradient max
        ratio_thresh_delta = get_ratio_thresh_delta(deltas)

        # Apply deltas
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        new_weights = []
        for layer_weights, layer_deltas in zip(model.get_weights(), ratio_thresh_delta):
            new_weights.append(layer_weights + layer_deltas)
        model.set_weights(new_weights)

        # Save final model
        model.save(MODEL_SAVE_PATH)

        #update callback
        callback.node = -1 
        #callback.on_epoch_end(iteration+1)

        metrics = model.evaluate(*test_data)
        with open("executions/training_log_mlp.csv", 'a') as f:
            f.write(','.join([str(val) for val in list(metrics)]))
        del model
        print(iteration, " model trained")

    callback.save(config['result_path'])
