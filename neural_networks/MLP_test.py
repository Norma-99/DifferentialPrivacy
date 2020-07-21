#This will import ADULT dataset and create an MLP to train the Dataset
import pickle
import tensorflow as tf
import numpy as np
import random

EPOCHS = 5 

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":

    # Load data
    print("Loading data")
    x_test, y_test = load_data('datasets/extended/validation/val_dataset.pickle')
    x_train, y_train = load_data('datasets/extended/test/split1/test_dataset.pickle')

    # Create network
    model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(74,)),
    tf.keras.layers.Dense(640, activation='relu', input_shape=(87,)),
    tf.keras.layers.Dense(640, activation='relu'), # 2/3 input + output
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train network
    print("training network")
    model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

