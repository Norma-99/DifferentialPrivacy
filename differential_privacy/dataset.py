import pickle
import typing

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get(self):
        return self.x, self.y

    def get_split(index: int, split_count: int) -> Dataset:
        pass

    @staticmethod
    def from_file(path: str):
        with open(path, 'rb') as f:
            return Dataset(*pickle.load(f))