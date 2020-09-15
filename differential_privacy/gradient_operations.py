def gradient_calc(initial, final):
    iteration_deltas = []
    for inital_layer_weights, final_layer_weights in zip(initial, final):
        iteration_deltas.append(final_layer_weights - inital_layer_weights)
    return iteration_deltas

def gradient_apply(weights, gradient):
    pass


    