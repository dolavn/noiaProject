import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from itertools import product
import scipy.io


def relu_activation_fp(z):
    return np.maximum(z, 0)


def relu_activation_grad(z):
    ans = z
    for i, j in product(*(range(dim) for dim in z.shape)):
        ans[i][j] = 1 if z[i][j] >= 0 else 0
    return ans


def tanh_activation_fp(z):
    return np.tanh(z)


def tanh_activation_grad(z):
    return 1 - np.tanh(z) ** 2


def get_mu(w, x, b, num_of_labels):
    mus = [np.dot(x.T, w[i])+b[i] for i in range(num_of_labels)]
    mu = np.array([max([m[i] for m in mus]) for i in range(mus[0].shape[0])])
    return mu


def create_c_matrix(y):
    c_vec = []
    l = y.shape[1]
    for val in range(l):
        c_vec.append([1 if elem[val] == 1 else 0 for elem in y])
    return np.array(c_vec).T


def sm_activation_fp(z):
    m = z.T.shape[1]
    ans = z
    for i in range(m):
        ans[i] = ans[i] - max(ans[i])
        ans[i] = np.exp(ans[i])
        ans[i] = ans[i]/sum(ans[i])
    return ans


def get_shapes(x, W, y, b):
    n, m = x.shape
    if W.shape[0] != n:
        raise BaseException("Dimensions incorrect")
    l = W.shape[1]
    if y.shape[1] != m:
        raise BaseException("Dimensions incorrect")
    if y.shape[0] != l:
        raise BaseException("Dimensions incorrect")
    if b.shape[0] != l:
        raise BaseException("Dimensions incorrect")
    return m, n, l


def sm_activation_gp(x, w, b, y):
    m, n, l = get_shapes(x, w, y, b)
    #print('m:{}\n n:{}\n l:{}'.format(m, n, l))
    c = create_c_matrix(y)
    #mu = get_mu(w, x, b, l)
    mu = 0
    sum_all = sum([np.exp(np.dot(w.T[i].T, x)+b[i]-mu) for i in range(l)])
    cis = [(np.exp(np.dot(w.T[i].T, x)+b[i]-mu)/sum_all)-c[i] for i in range(l)]
    cis = [1/len(x)*np.dot(x, elem) for elem in cis]
    cis = [elem.reshape(1, -1) for elem in cis]
    ans = np.concatenate(cis, axis=0)
    bs = [np.exp(np.dot(w.T[i].T, x)+b[i]-mu)/sum_all for i in range(l)]
    bs = [np.dot(curr, np.ones(len(curr))) for curr in bs]
    bs = [curr - sum(c[i]) for i, curr in enumerate(bs)]
    bs = np.array([(1/len(x))*curr for curr in bs])
    return ans, bs


def sm_activation_gx(x, W, y, b):
    m, n, l = get_shapes(x, W, y, b)
    c = create_c_matrix(y)
    v_sum = sum([np.exp(np.dot(W.T[i].T, x)) for i in range(l)])
    v_sum = np.matlib.repmat(v_sum, l, 1)
    ans = np.dot(W.T, x)/v_sum
    ans = ans - c
    ans = np.dot(W, ans)
    ans = ans/m
    return ans


RELU_ACTIVATION = (relu_activation_fp, relu_activation_grad)
TANH_ACTIVATION = (tanh_activation_fp, tanh_activation_grad)
SOFTMAX_ACTIVATION = (sm_activation_fp, sm_activation_gp)

ACTIVATION_FUNCTIONS = [RELU_ACTIVATION, TANH_ACTIVATION]


class Layer:

    def __init__(self, input_dim, output_dim, activation, softmax_layer=False):
        if softmax_layer and activation is not None:
            raise BaseException("Can't give another activation function to softmax layer")
        if activation is not None and activation not in ACTIVATION_FUNCTIONS:
            raise BaseException("Invalid activation function")
        if softmax_layer:
            self._forward_pass, self._gradient = SOFTMAX_ACTIVATION
        if activation:
            self._forward_pass, self._gradient = activation
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weights = np.random.rand(input_dim, output_dim)
        self._bias = np.random.rand(output_dim)
        self._x = None
        self._z = None
        self._a = None
        self._delta = None
        self._g = None
        self._b = None
        self._batch_size = None
        self._theta_grad = None
        self._jacobian = None
        self._x_grad = None
        self._all_diags = None
        self._softmax_layer = softmax_layer


    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_delta(self):
        return self._delta

    def forward_pass(self, input):
        if self._input_dim != input.shape[0]:
            print('received:{}\nactual:{}'.format(input.shape[0], self._input_dim))
            raise BaseException("Invalid dimensions")
        self._x = input
        self._batch_size = input.shape[1]
        self._z = np.dot(input.T, self._weights)+np.matlib.repmat(self._bias, input.shape[1], 1)
        self._a = self._forward_pass(self._z)
        return self._a

    def back_propagation(self, next_layer=None):
        if self._softmax_layer:
            raise BaseException("Can not be performed on a softmax layer")
        if not next_layer:
            raise BaseException("No next layer")
        else:
            self.calc_jacobian()
            self._delta = np.dot(self._x_grad.T, next_layer.get_delta())
            self._g = np.dot(self._jacobian.T, next_layer.get_delta()).reshape(*self._weights.shape)
            self._b = np.dot(self._all_diags.T, next_layer.get_delta()).reshape(*self._bias.shape)
            self._theta_grad = np.concatenate((self._g.flatten(), self._b))

    def calc_jacobian(self):
        all_grads_p = []
        all_grads_x = []
        all_diags = []
        for i in range(self._batch_size):
            curr_sigma = self._gradient(np.atleast_2d(self._z[i]))[0]
            diag = np.diag(curr_sigma)
            t = np.tensordot(self._x.T[i], np.identity(self._output_dim), axes=0)
            xt = np.dot(diag, self._weights.T)
            t = t.reshape(self._input_dim * self._output_dim, self._output_dim).T
            all_grads_p.append(np.dot(diag, t))
            all_grads_x.append(xt)
            all_diags.append(diag)
        all_grads_p = np.array(all_grads_p).reshape(self._batch_size * self._output_dim,
                                                    self._input_dim * self._output_dim)
        self._x_grad = block_diag(*all_grads_x)
        self._all_diags = np.array(all_diags).reshape(self._batch_size * self._output_dim,
                                                self._output_dim)
        self._jacobian = all_grads_p

    def calc_softmax_grad(self, labels):
        if not self._softmax_layer:
            raise BaseException("Can only be performed on a softmax layer")
        self._delta = sm_activation_gx(self._x, self._weights, labels, self._bias).flatten()
        w_grad, b_grad = self._gradient(self._x, self._weights, self._bias, labels)
        self._theta_grad = np.concatenate((w_grad.flatten(), b_grad))

    def set_weights(self, weights):
        update_w = np.array(weights[: self._input_dim*self._output_dim]).reshape(self._input_dim,
                                                                                 self._output_dim)
        #update_b = np.array(weights[self._input_dim*self._output_dim:])
        self._weights = update_w
        #self._bias = update_b

    def inc_weights(self, weights):
        update_w = np.array(weights[: self._input_dim*self._output_dim]).reshape(self._input_dim,
                                                                                 self._output_dim)
        update_b = np.array(weights[self._input_dim*self._output_dim:])
        self._weights = self._weights + update_w
        self._bias = self._bias + update_b

    def get_param_num(self):
        return self._input_dim*self._output_dim+self._output_dim

    def get_theta_grad(self):
        return self._theta_grad

    def get_jacobian(self):
        return self._jacobian


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

    def predict(self, input):
        return self.forward_pass(input)

    def get_grad(self, input, labels):
        self.forward_pass(input)
        self.back_propagation(labels)
        return np.concatenate([layer.get_theta_grad() for layer in self._layers])

    def forward_pass(self, input):
        if self._input_dim != input.shape[0]:
            raise BaseException("Invalid dimensions, expected {} but got {}".format(self._input_dim,
                                                                                    input.shape[0]))
        curr_input = input
        for layer in self._layers:
            curr_input = layer.forward_pass(curr_input).T
        return curr_input

    def back_propagation(self, labels):
        self._layers[-1].calc_softmax_grad(labels)
        indices = [i for i in range(len(self._layers)-1)]
        for ind in indices[::-1]:
            next_layer = self._layers[ind+1]
            curr_layer = self._layers[ind]
            curr_layer.back_propagation(next_layer)
            #curr_layer.update_weights(alpha)

    def inc_weights(self, new_weights):
        last_ind = 0
        for layer in self._layers:
            curr_ind = layer.get_input_dim()*layer.get_output_dim()+layer.get_output_dim()
            curr_ind = curr_ind + last_ind
            curr_theta = np.array(new_weights[last_ind: curr_ind])
            layer.inc_weights(curr_theta)
            last_ind = curr_ind

    def calc_error(self, x, y):
        output = self.forward_pass(x)
        return np.linalg.norm(y-output)

    def get_param_num(self):
        return sum([layer.get_param_num() for layer in self._layers])

    def get_layer(self, ind):
        if ind >= len(self._layers):
            raise BaseException("Index out of bounds")
        return self._layers[ind]

#TESTING


if __name__ == '__main__':
    X = np.array([[2, 3, 1], [1, 5, 2], [4, 2, 3], [1, 4, 1], [2, 1, 4]]).T
    Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]).T
    W = np.array([[0.5, 0.5, 1], [0.2, 0, 1]]).T
    b = np.array([0, 0])
    g = sm_activation_gx(X, W, Y, b)
    mat = scipy.io.loadmat('SwissRollData.mat')
    labels = mat['Ct']
    training = mat['Yt']
    x = training
    y = labels
    """
    t = 50
    x = np.zeros((t**2, 2))
    y = np.zeros((t**2, 2))

    curr = 0
    for i1, i2 in product(np.linspace(-1.5, 1.5, t), np.linspace(-1.5, 1.5, t)):
        x[curr][0] = i1
        x[curr][1] = i2
        if i2 < 0:
            y[curr][1] = 1
        else:
            y[curr][0] = 1
        curr = curr + 1
    x = x.T
    y = y.T
    """
    n = Network()
    n.add_layer(Layer(2, 5, RELU_ACTIVATION))
    n.add_layer(Layer(5, 2, None, softmax_layer=True))
    errors = []
    alpha = 0.05
    batches = np.random.permutation(range(y.shape[1]))
    batch_size = 100
    curr_ind = 0
    print(n.calc_error(x, y))
    for i in range(100):
        curr_batch = batches[curr_ind: curr_ind+batch_size]
        curr_ind = curr_ind+batch_size
        batch_x = np.array([x.T[ind] for ind in curr_batch]).T
        batch_y = np.array([y.T[ind] for ind in curr_batch]).T
        g = n.get_grad(batch_x, batch_y)
        g = g*alpha
        n.inc_weights(-g)
        #alpha = alpha*0.999
        errors.append(n.calc_error(x, y))
    #plt.plot(range(len(errors)), errors)
    #plt.show()
    #exit()
    print(errors[-1])
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-1.5, 1.5, 100)
    image = np.zeros((100, 100))
    image2 = np.zeros((100, 100))
    for (i1, x_i), (i2, y_i) in product(enumerate(x_range),
                                        enumerate(y_range)):
        vec = np.atleast_2d(np.array([x_i, y_i]))
        label = n.forward_pass(vec.T)
        image[i1][i2] = label[0]
    plt.imshow(image, extent=[-1.5, 1.5, -1.5, 1.5])
    coord_x_pos = [x.T[ind][0] for ind in range(y.shape[1]) if y.T[ind][0] == 1]
    coord_y_pos = [x.T[ind][1] for ind in range(y.shape[1]) if y.T[ind][0] == 1]
    coord_x_neg = [x.T[ind][0] for ind in range(y.shape[1]) if y.T[ind][0] == 0]
    coord_y_neg = [x.T[ind][1] for ind in range(y.shape[1]) if y.T[ind][0] == 0]
    plt.scatter(coord_x_pos, coord_y_pos, alpha=0.2)
    plt.scatter(coord_x_neg, coord_y_neg, alpha=0.2)
    plt.show()
    exit()
