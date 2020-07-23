#This will import ADULT dataset and use Hyperband to decide which model is the most optimal for the dataset
from functools import partial

import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import kerastuner as kt

EPOCHS = 5 
HIDDEN = 3

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def model_builder(input_shape, hp):
    model = tf.keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))

    for i in range(HIDDEN):
        hp_filters = hp.Int(f'filters{i}', min_value=128, max_value=2056, step=8)
        model.add(Dense(units = hp_filters, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    model.compile(optimizer = Adam(learning_rate = hp_learning_rate),
                loss = 'binary_crossentropy', 
                metrics = ['accuracy'])
    return model

if __name__ == "__main__":

    # Load data
    print("Loading data")
    x_test, y_test = load_data('datasets/extended/validation/val_dataset.pickle')
    x_train, y_train = load_data('datasets/extended/test/split1/test_dataset.pickle')

    tuner = kt.Hyperband(
    partial(model_builder, (x_train.shape[1:])),
    objective = 'val_accuracy',
    max_epochs = EPOCHS,
    directory = 'result_dir',
    project_name = 'test2_from128_to2056_in8')

    tuner.search(
        x_train,
        y_train,
        epochs = EPOCHS,
        validation_data = (x_test, y_test))

    models = tuner.get_best_models(num_models=5)
    models[0].summary()
    models[1].summary()
    models[2].summary()
    models[3].summary()
    models[4].summary()

