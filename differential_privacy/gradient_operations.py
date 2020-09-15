def gradient_calc(initial, final):
    iteration_deltas = []
    for inital_layer_weights, final_layer_weights in zip(initial, final):
        iteration_deltas.append(final_layer_weights - inital_layer_weights)
    return iteration_deltas

#mirar
def gradient_apply(weights, gradient):
    new_weights = []
    for layer_weights, layer_gradient in zip(weights, gradient):
            new_weights.append(layer_weights + layer_gradient)
    return new_weights
 