import numpy as np
import numpy.matlib


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


def get_mu(w, x, b, num_of_labels):
    mus = [np.dot(x.T, w[i])+b[i] for i in range(num_of_labels)]
    mu = np.array([max([m[i] for m in mus]) for i in range(mus[0].shape[0])])
    return mu


def create_c_matrix(y):
    c_vec = []
    l = y.shape[1]
    for val in range(l):
        c_vec.append([1 if elem[val] == 1 else 0 for elem in y])
    return np.array(c_vec)


def sm_activation_fp(x, W, b):
    m = x.shape[0]
    num_of_labels = W.shape[0]
    mu = get_mu(W, x, b, num_of_labels)
    sum_all = sum([np.exp(np.dot(x.T, W[i])+b[i]-mu) for i in range(num_of_labels)])
    mu_mat = np.matlib.repmat(mu, m, 1).T
    ans = np.dot(x.T, W)+b-mu_mat
    ans = np.exp(ans) / np.matlib.repmat(sum_all, m, 1).T
    return ans


def sm_activation_gp(x, w, b, y):
    c = create_c_matrix(y)
    num_of_labels = y.shape[1]
    mu = get_mu(w, x, b, num_of_labels)
    sum_all = sum([np.exp(np.dot(x.T, w[i])+b[i]-mu) for i in range(num_of_labels)])
    cis = [(np.exp(np.dot(x.T, w[i])+b[i]-mu)/sum_all)-c[i] for i in range(num_of_labels)]
    cis = [1/len(x)*np.dot(x, elem) for elem in cis]
    cis = [elem.reshape(1, -1) for elem in cis]
    ans = np.concatenate(cis, axis=0)
    bs = [np.exp(np.dot(x.T, w[i]) + b[i]-mu)/sum_all for i in range(num_of_labels)]
    bs = [np.dot(curr, np.ones(len(curr))) for curr in bs]
    bs = [curr - sum(c[i]) for i, curr in enumerate(bs)]
    bs = np.array([(1/len(x))*curr for curr in bs])
    return np.concatenate((ans.flatten, bs))


def get_shapes(x, W, y, b):
    n, m = x.shape
    if W.shape[0] != n:
        raise BaseException("Dimensions incorrect")
    l = W.shape[1]
    if y.shape[0] != m:
        raise BaseException("Dimensions incorrect")
    if y.shape[1] != l:
        raise BaseException("Dimensions incorrect")
    if b.shape[0] != l:
        raise BaseException("Dimensions incorrect")
    return m, n, l


def sm_activation_gx(x, W, y, b):
    m, n, l = get_shapes(x, W, y, b)
    c = create_c_matrix(y)
    v_sum = sum([np.exp(np.dot(W[i].T, x)) for i in range(n)])
    v_sum = np.matlib.repmat(v_sum, l, 1)
    ans = np.dot(W.T, x)/v_sum
    ans = ans - c
    ans = np.dot(W, ans)
    ans = ans/m
    return ans


RELU_ACTIVATION = (relu_activation_fp, relu_activation_gp, relu_activation_gx)
TANH_ACTIVATION = (tanh_activation_fp, tanh_activation_gp, tanh_activation_gx)
SOFTMAX_ACTIVATION = (sm_activation_fp, sm_activation_gp, sm_activation_gx)

ACTIVATION_FUNCTIONS = [RELU_ACTIVATION, TANH_ACTIVATION]


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
        if self._input_dim == -1:
            self._input_dim = layer.get_input_dim()
        if layer.get_input_dim() != self._output_dim != -1:
            raise BaseException("Invalid dimensions")
        self._layers.append(layer)
        self._output_dim = layer.get_output_dim()

    def forward_pass(self, input):
        if self._input_dim != input.shape[0]:
            raise BaseException("Invalid dimensions")


#TESTING


if __name__ == '__main__':
    X = np.array([[1, 2], [2, 2], [2, 3]]).T
    Y = np.array([[1, 0], [0, 1], [0, 1]])
    W = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    g = sm_activation_gx(X, W, Y, b)
    print(g)
    print(sm_activation_fp(X, W, b))
    n = Network()
