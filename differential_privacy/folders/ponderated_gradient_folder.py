from .gradient_folder import GradientFolder

class PonderatedGradientFolder(GradientFolder):
    def __init__(self):
        GradientFolder.__init__(self)
        self.network = None

    def fold(self, neural_network, gradients):
        self.network = neural_network
        grades = map(self._evaluate_gradient, gradients)
        grade_sum = sum(grades)
        grades = [grade / grade_sum for grade in grades]
        return gradient_weighted_mean(gradients, grades) #TODO: Implement


    def _evaluate_gradient(self, gradient):
        network = self.network.clone()
        network.apply_gradient(gradient) # TODO: Implement
        results = network.evaluate(self.generalization_dataset)
        return results['acc']