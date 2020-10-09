import math
from . import Dataset


class SplittableDataset(Dataset):
    def __init__(self, x, y):
        Dataset.__init__(self, x, y)

    def get_split(self, index: int, split_count: int) -> Dataset:
        split_size = math.floor(len(self.x) / split_count)
        start = index * split_size
        return Dataset(self.x[start:start + split_size], self.y[start:start + split_size])

    @staticmethod
    def from_dataset(dataset: Dataset):
        return SplittableDataset(*dataset.get())