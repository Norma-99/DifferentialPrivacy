import re
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
        logs = rekey(logs)
        self.data.append({
            'iteration': self.iteration,
            'epoch': epoch,
            'fog_node': self.fog_node,
            'device': self.device,
            **logs
        })

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path)


def rekey(logs):
    new_logs = dict()
    for key in logs.keys():
        if re.search('_[0-9]', key):
            new_logs[key[:-2]] = logs[key]
        else:
            new_logs[key] = logs[key]
    return new_logs