from differential_privacy.encryptor import Encryptor

class ExperimentEncryptor(Encryptor):
    def encript(self, gradients):
        return gradients[0]
