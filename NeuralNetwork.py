import numpy as np

def relu_activation():
    pass

def tanh_activation():
    pass


class Layer:

    def __init__(self, input_dim, output_dim, activation):
        if activation not in [relu_activation, tanh_activation]:
            raise BaseException("Invalid activation function")
        self._activation = activation
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weights = np.random.rand(input_dim, output_dim)
        self._bias = np.random.rand(output_dim)

    def forward_pass(self, input):
        if self._input_dim != input.shape[0]:
            raise BaseException("Bad dimensions")
        return self._activation(np.dot(input, self._weights)+self._bias)


class Network:

    def __init__(self):
        self._layers = []


