import pickle
import math


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get(self):
        return self.x, self.y

    @staticmethod
    def load_validation(path):
        return Dataset(*load_data(path))
    
    @staticmethod
    def load_splits(path, num_splits):
        x, y = load_data(path)
        split_size = math.floor(len(x) / num_splits)
        for split_id in range(num_splits):
            start = split_id * split_size
            end = start + split_size
            yield Dataset(x[start:end], y[start:end])


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)