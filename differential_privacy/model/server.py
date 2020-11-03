from . import NetworkComponent, NeuralNetwork


class Server(NetworkComponent):
    def __init__(self, neural_network:NeuralNetwork):
        NetworkComponent.__init__(self)
        self.neural_network = neural_network

    def on_data_receive(self, data: dict):
        if 'gradient' in data:
            self.neural_network.apply_gradient(data['gradient'])
        else:
            self.send({'neural_network': self.neural_network.clone()}, data['origin'])

    def _process_dataset_request(data: dict):
        pass

    def _process_gradient(data: dict):
        pass