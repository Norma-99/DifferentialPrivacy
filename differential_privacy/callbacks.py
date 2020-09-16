import pandas as pd
from tensorflow.keras.callbacks import Callback

class ReportCallback(Callback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.iteration = 0
        self.fog_node = 0
        self.device = 0
        self.data = []
    
    def on_epoch_end(self, epoch, logs=None):
        self.data.append({
            'iteration': self.iteration,
            'fog_node': self.fog_node,
            'device': self.device,
            **logs
        })
    
    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path)
