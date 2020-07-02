#This will import ADULT dataset and create an MLP to train the Dataset
#This is MLP with the geometric mean delta (later try the threshold)
import pickle
import tensorflow as tf
import numpy as np
import random

NODES = 7
ITERATIONS = 10
EPOCHS = 1
MODEL_SAVE_PATH = 'temp_mlp.h5'


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
    # Create network
    model = tf.keras.Sequential([
    #(87,) or (74,)
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(74,)),
    tf.keras.layers.Dense(810, activation='relu'), # 2/3 input + output
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.save(MODEL_SAVE_PATH)
    del model

    test_data = load_data('datasets/reduced/validation/val_dataset.pickle')

    for iteration in range(int(ITERATIONS)):
        deltas = []
        for i in range(NODES):
            
            # Copy net
            print("Copying net")
            iteration_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            initial_weights = iteration_model.get_weights()
            
            # Load data
            print("Loading data")
            x_train, y_train = load_data('datasets/reduced/test/split7/datasplit%04d.pickle' % (i%1))
            #x_train, y_train = load_data('datasets/reduced/test/split1/test_dataset.pickle')

            # Train network
            print("training network")
            iteration_model.fit(x_train, y_train, epochs=EPOCHS)

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
        
        metrics = model.evaluate(*test_data)
        with open("executions/training_log_mlp.csv", 'a') as f:
            f.write(','.join([str(val) for val in list(metrics)]))
        del model
        print(iteration, " model trained")
