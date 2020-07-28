#This will import ADULT dataset and use Hyperband to decide which model is the most optimal for the dataset
import argparse
from functools import partial

import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import kerastuner as kt

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def model_builder(input_shape, config, hp):
    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))

    for i in range(config['hidden_layers']):
        hp_filters = hp.Int(
            f'filters{i}',
            min_value=config['filters_min_value'],
            max_value=config['filters_max_value'],
            step=config['filters_step'])
        model.add(Dense(units = hp_filters, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values = config['learning_rates'])
    model.compile(optimizer = Adam(learning_rate = hp_learning_rate),
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    # Load data
    print("Loading data")
    x_train, y_train = load_data(args.config['train_dataset'])
    x_test, y_test = load_data(args.config['val_dataset'])

    tuner = kt.Hyperband(
    partial(model_builder, x_train.shape[1:], args.config),
    objective = 'val_accuracy',
    max_epochs = args.config['epochs'],
    directory = 'result_dir',
    project_name = args.config['project_name'])

    tuner.search(
        x_train,
        y_train,
        epochs = args.config['epochs'],
        validation_data = (x_test, y_test))

    models = tuner.get_best_models(num_models=5)
    models[0].summary()
    models[1].summary()
    models[2].summary()
    models[3].summary()
    models[4].summary()

