import numpy as np

def gradient_calc(initial:list, final:list):
    return [np.subtract(layer_final_weights, layer_inital_weight)
            for layer_final_weights, layer_inital_weight in zip(final, initial)]
    

def gradient_apply(weights:list, gradient:list):
    return [np.add(layer_weights, layer_gradient)
            for layer_weights, layer_gradient in zip(weights, gradient)]


def gradient_median(gradients:list):
    res = []
    for layer_index in range(len(gradients[0])):
        res.append(np.median([g[layer_index] for g in gradients], axis=0))
    return res