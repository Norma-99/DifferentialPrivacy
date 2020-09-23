from tensorflow.keras.models import clone_model
from differential_privacy.experiment_params import ExperimentParams
from differential_privacy.fog_node import FogNode
from differential_privacy.dataset import Dataset
from differential_privacy.gradient_operations import gradient_median, gradient_apply, gradient_calc


class Server:
    def __init__(self, experiment_params:ExperimentParams):
        self.model = experiment_params.model
        num_devices = experiment_params.devices // experiment_params.fog_nodes
        self.fog_nodes = [FogNode(
            experiment_params.splits[i:i+num_devices],
            experiment_params.encryptor,
            experiment_params.model_manager) 
            for i in range(0, experiment_params.devices, num_devices)]
        self.val_data = experiment_params.validation
        self.iterations = experiment_params.iterations
        self.log_callback = experiment_params.model_manager.model_fitter.callback
        self.model_manager = experiment_params.model_manager

    def train(self):
        for iteration in range(self.iterations):
            self.log_callback.iteration = iteration
            gradients = [fog_node.get_enc_gradient(self.model_manager.clone_model(self.model)) for fog_node in self.fog_nodes] 
            gradient = gradient_median(gradients)
            self.model.set_weights(gradient_apply(self.model.get_weights(), gradient))
            #Incorporación Norma
            self.model_manager.evaluate(self.model, self.val_data)
            #self.model.evaluate(*self.val_data.get(), verbose=0)
        self.log_callback.save()