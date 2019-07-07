import numpy as np


def relu_activation(x):
    return x


def tanh_activation(x):
    return x


class Layer:

    def __init__(self, input_dim, output_dim, activation):
        if activation not in [relu_activation, tanh_activation]:
            raise BaseException("Invalid activation function")
        self._activation = activation
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weights = np.random.rand(input_dim, output_dim)
        self._bias = np.random.rand(output_dim)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def forward_pass(self, input):
        if self._input_dim != input.shape[0]:
            raise BaseException("Invalid dimensions")
        return self._activation(np.dot(input, self._weights)+self._bias)


class Network:

    def __init__(self):
        self._input_dim = -1
        self._output_dim = -1
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)
        if self._input_dim == -1:
            self._input_dim = layer.get_input_dim()
        self.output_dim = layer.get_output_dim()

    def forward_pass(self, input):
        if self._input_dim != input.shape[0]:
            raise BaseException("Invalid dimensions")


#TESTING


if __name__ == '__main__':
    l = Layer(2, 5, relu_activation)
    a = np.array([1, 2, 1])
