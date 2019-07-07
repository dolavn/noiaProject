import numpy as np


def relu_activation_fp(x, W, b):
    return x


def relu_activation_gp(x, W, b):
    return x


def relu_activation_gx(x, W, b):
    return x


def tanh_activation_fp(x, W, b):
    return x


def tanh_activation_gp(x, W, b):
    return x


def tanh_activation_gx(x, W, b):
    return x


def sm_activation_fp(x, W, b):
    return x


def sm_activation_gp(x, W, b):
    return x


def sm_activation_gx(x, W, b):
    return x


RELU_ACTIVATION = (relu_activation_fp, relu_activation_gp, relu_activation_gx)
TANH_ACTIVATION = (tanh_activation_fp, tanh_activation_gp, tanh_activation_gx)
SOFTMAX_ACTIVATION = (sm_activation_fp, sm_activation_gp, sm_activation_gx)

ACTIVATION_FUNCTIONS = [RELU_ACTIVATION, TANH_ACTIVATION, SOFTMAX_ACTIVATION]


class Layer:

    def __init__(self, input_dim, output_dim, activation):
        if activation not in ACTIVATION_FUNCTIONS:
            raise BaseException("Invalid activation function")
        self._forward_pass, self_gradient_params, self._gradient_x = activation
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
        return self._forward_pass(input, self._weights, self._bias)


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
