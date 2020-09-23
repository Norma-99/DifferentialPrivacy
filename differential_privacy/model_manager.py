from tensorflow.keras.models import clone_model
from tensorflow.keras.metrics import AUC, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from differential_privacy.callbacks import ReportCallback
from differential_privacy.dataset import Dataset

class ModelCloner:
    def __init__(self, config:dict):
        self.optimizer = config['optimizer']
        self.loss = config['loss']
        self.metrics = [
            'accuracy', 
            AUC(),
            FalseNegatives(),
            FalsePositives(), 
            TrueNegatives(), 
            TruePositives()]

    def clone_model(self, model):
        cp_model = clone_model(model)
        cp_model.set_weights(model.get_weights())
        cp_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return cp_model


class ModelFitter:
    def __init__(self, config:dict): 
        self.epochs = config['epochs']
        self.callback = ReportCallback(config['result_path'])

    def fit(self, model, dataset:Dataset):
        model.fit(*dataset.get(), epochs=self.epochs, callbacks=[self.callback])

    #Incorporación Norma
    def evaluate(self, model, dataset:Dataset):
        res = model.evaluate(*dataset.get(), callbacks=[self.callback], verbose=0, return_dict=True)
        self.callback.on_epoch_end(-1, res)

class ModelManager:
    def __init__(self, config:dict):
        self.model_cloner = ModelCloner(config)
        self.model_fitter = ModelFitter(config)

    def clone_model(self, model):
        return self.model_cloner.clone_model(model)
    
    def fit(self, model, fog_node, device):
        self.model_fitter.callback.fog_node = fog_node.id
        self.model_fitter.callback.device = device.id
        self.model_fitter.fit(model, device.dataset)
    
    #Incorporación Norma
    def evaluate(self, model, dataset):
        self.model_fitter.evaluate(model, dataset)