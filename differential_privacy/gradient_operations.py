import numpy as np

def gradient_calc(initial, final):
    return np.subtract(final, initial)
    

def gradient_apply(weights, gradient):
    return np.add(weights, gradient)


def gradient_median(gradients:list):
    return np.median(gradients, axis=0)
    
 