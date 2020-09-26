import numpy as np
import random
from differential_privacy.encryptor import Encryptor

class ExperimentEncryptor(Encryptor):
    def encript(self, gradients):
        ratio_thresh_gradient = []
        nodes = range(0, len(gradients))
        layers = range(0, len(gradients[0]))
        for layer in layers:
            layer_weights = list()
            layer_max_gradient = gradients[0][layer]
            for node in nodes:
                layer_weights.append(gradients[node][layer])
                layer_max_gradient = np.maximum(layer_max_gradient, gradients[node][layer])
            #Define a random number between median and a 50% between median and maximun
            layer_median_gradient = np.median(layer_weights, axis=0)
            layer_difference_gradient = (layer_max_gradient - layer_median_gradient)/2
            ratio_thresh_gradient.append(random.uniform(layer_median_gradient, layer_median_gradient + layer_difference_gradient))
        return ratio_thresh_gradient
