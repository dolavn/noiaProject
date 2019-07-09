import numpy as np
import numpy.matlib
from itertools import product


def relu_activation_fp(z):
    return np.maximum(z, 0)



def relu_activation_gp(x, W, b):
    dg_elem = np.append(x, np.ones(x.shape[1]))
    argument = np.dot(W, x) + (np.matlib.repmat(b, x.shape[1], 1)).T
    dg = []
    for arg1 in argument:
        for arg2 in arg1:
            if arg2 > 0:
                dg = np.append(dg, dg_elem)
            else:
                dg = np.append(dg, np.zeros(dg_elem.shape))
    return dg


def relu_activation_gx(x, W, b):
    dg_elem = W
    argument = np.dot(W, x) + (np.matlib.repmat(b, x.shape[1], 1)).T
    dg = []
    for arg1 in argument:
        for arg2 in arg1:
            if arg2 > 0:
                dg = np.append(dg, dg_elem)
            else:
                dg = np.append(dg, np.zeros(dg_elem.shape))
    return dg


def relu_activation_grad(z):
    ans = z
    for i, j in product(*(range(dim) for dim in z.shape)):
        ans[i][j] = 1 if z[i][j] >=0 else 0
    return ans


def tanh_activation_fp(z):
    return np.tanh(z)



def tanh_activation_grad(z):
    return 1 - np.tanh(z) ** 2


def tanh_activation_gx(x, W, b):
    argument = np.dot(W, x) + (np.matlib.repmat(b, x.shape[1], 1)).T
    return np.dot(W.T, (1 - np.tanh(argument) ** 2)).flatten()



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
    m = x.T.shape[0]
    num_of_labels = W.shape[0]
    ans = np.dot(x.T, W)+np.matlib.repmat(b, m, 1)
    for i in range(m):
        ans[i] = ans[i] - max(ans[i])
        ans[i] = np.exp(ans[i])
        ans[i] = ans[i]/sum(ans[i])
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
        return self._forward_pass(np.dot(input,
                                         self._weights)+np.matlib.repmat(self._bias,
                                                                         input.shape[1], 1))


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
    X = np.array([[2, 3], [5, 2], [2, 3]]).T
    Y = np.array([[1, 0], [0, 1], [0, 1]])
    W = np.array([[0.5, 0.5], [0, 1]])
    b = np.array([0, 0])
    g = sm_activation_gx(X, W, Y, b)
    print(np.dot(X.T, W))
    print(g)
    print(sm_activation_fp(X, W, b))
    n = Network()

    exit()

    #Shlomit's testing code
    a = np.array([1, 2, 1])
    x = np.array([[-5, -2], [2, -2], [-1, 0], [-1, 1], [0, 2], [1, 4], [2, 8], [1, 2], [2, 4], [1, 9]]).T
    #y = np.array([2, 3, 0, 1, 1, 2, 3, 0, 3, 2])
    W = np.array([[0.2, 1], [0.1, 0], [1, 0], [0.1, 0]])
    print(x.shape)
    print(W.shape)
    b = np.array([1, 1, 1, 1])
    dgp=relu_activation_gp(x, W, b)
    print(dgp[60:90])
    dgx = relu_activation_gx(x, W, b)
    print(dgx[16:24])
    print(tanh_activation_gp(x, W, b))
    print(tanh_activation_gx(x, W, b))




