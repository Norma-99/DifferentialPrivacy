import numpy
from differential_privacy.encryptor import Encryptor

class ExperimentEncryptor(Encryptor):
    def encript(self, gradients):
        mean_gradient = []
        nodes = range(0, len(gradients))
        layers = range(0, len(gradients[0]))
        for layer in layers:
            layer_weights = list()
            for node in nodes:
                layer_weights.append(deltas[node][layer])
            mean_gradient.append(np.mean(layer_weights, axis=0))
        return mean_gradient
