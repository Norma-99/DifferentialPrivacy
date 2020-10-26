import pickle
import typing

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def get(self):
        return self.x, self.y
    
    def get_split(index: int, split_count: int):
        pass

    #Propongo hacer un método get_generalisation_fragment que te de el split de cada dataset
    def get_generalisation_fragment(fraction: float):
        fragment_size = len(self.x) // fraction
        return Dataset(x[:fragment_size], y[:fragment_size])

    @staticmethod
    def from_file(path: str):
        with open(path, 'rb') as f:
            return Dataset(*pickle.load(f))