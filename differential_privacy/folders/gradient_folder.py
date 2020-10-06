class GradientFolder:
    def __init__(self):
        self.generalization_dataset = GeneralizationDataset()

    def add_subdataset(self, dataset):
        self.generalization_dataset.concatenate(dataset)

    def fold(self, neural_network, gradients):
        raise NotImplementedError()