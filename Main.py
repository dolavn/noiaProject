import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io
from Optimizer import stochastic_gradient_descent
from NeuralNetwork import Network, TANH_ACTIVATION, RELU_ACTIVATION, Layer
from itertools import product


def get_mu(w, x, b, num_of_labels):
    mus = [np.dot(x.T, w[i])+b[i] for i in range(num_of_labels)]
    mu = np.array([max([m[i] for m in mus]) for i in range(mus[0].shape[0])])
    #mu = np.zeros(*mu.shape)
    return mu


def create_c_vecs(y, num_of_labels):
    ans = []
    for val in range(num_of_labels):
        ans.append([1 if elem[val] == 1 else 0 for elem in y])
    return np.array(ans)


def softmax_obj(x, y, w, b):
    num_of_labels = y.shape[1]
    c = create_c_vecs(y, num_of_labels)
    mu = get_mu(w, x, b, num_of_labels)
    sum_all = sum([np.exp(np.dot(x.T, w[i])+b[i]-mu) for i in range(num_of_labels)])
    val = -sum([np.dot(c[i].T, np.log(np.exp(np.dot(x.T, w[i])+b[i]-mu)/sum_all)) for i in range(num_of_labels)])/len(x)
    return val


def softmax_grad(x, y, w, b):
    num_of_labels = y.shape[1]
    c = create_c_vecs(y, num_of_labels)
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
    return ans, bs


def gradient_test(f, g, dim):
    """
    Performs the gradient test on f
    :param f: A function
    :param g: The gradient of f
    :param dim: The dimensions of the input to f
    :return: None
    """
    x = np.random.rand(*dim)
    d = np.random.rand(*dim)
    d = d/np.linalg.norm(d)
    lins = []
    quads = []
    epsilons = np.linspace(0, 0.001, 200)
    for epsilon in epsilons:
        lin = np.abs(f(x+epsilon*d)-f(x))
        print(lin)
        d_flat = d.flatten()
        g_flat = g(x).flatten()
        dot = np.dot(d_flat, g_flat)
        quad = np.abs(f(x+epsilon*d)-f(x)-epsilon*dot)
        lins.append(lin*50)
        quads.append(quad)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Gradient test')
    plt.plot(epsilons, lins)
    plt.plot(epsilons, quads)
    plt.legend(['$|f(x)+\\epsilon\\cdot d)-f(x)|$',
                '$|f(x+\\epsilon\\cdot d)-f(x)-\\epsilon d^{\\top}\\cdot grad(x)|$'])
    plt.savefig('gradient_test.png')



#x = np.array([[5, 2], [2, 2], [1, 0], [1, 1], [0, 2], [1, 4], [2, 8], [1, 2], [2, 4], [1, 9]]).T
#y = np.array([2, 3, 0, 1, 1, 2, 3, 0, 3, 2])
#w = np.array([[0.2, 1], [0.1, 0], [1, 0], [0.1, 0]])
#b = np.array([0, 0, 0, 0])


mat = scipy.io.loadmat('SwissRollData.mat')
labels = mat['Ct']
training = mat['Yt']

x = training
y = labels
w = np.random.rand(x.shape[0], y.shape[0])
b = np.random.rand(y.shape[0])


def obj(wb, indices):
    num_of_labels = y.shape[0]
    x_batch = np.concatenate([x.T[i] for i in indices]).reshape(*x.T[0].shape, len(indices))
    y_batch = np.array([y.T[i] for i in indices])
    w_range = x.shape[0]*num_of_labels
    curr_w = wb[:w_range].reshape(num_of_labels, x.shape[0])
    curr_b = wb[w_range:]
    return softmax_obj(x_batch, y_batch, curr_w, curr_b)


def grad(wb, indices):
    num_of_labels = y.shape[0]
    x_batch = np.concatenate([x.T[i] for i in indices]).reshape(*x.T[0].shape, len(indices))
    y_batch = np.array([y.T[i] for i in indices])
    w_range = x.shape[0]*num_of_labels
    curr_w = wb[:w_range].reshape(num_of_labels, x.shape[0])
    curr_b = wb[w_range:]
    wgrad, bgrad = softmax_grad(x_batch, y_batch, curr_w, curr_b)
    #print(wgrad)
    #print(bgrad)
    return np.concatenate((wgrad.flatten(), bgrad.T), axis=None)


def jacobian_test(f, j, dim):
    """
    Performs the jacobian test on f
    :param f: A function f
    :param j: A method that calculates M*v where v is a vector and M is the Jacobian
    :param dim: The dimensions of the input to f
    :return: None
    """
    x = np.random.rand(dim)
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.1])
    d = np.random.rand(dim)
    d = d/np.linalg.norm(d)
    lins = []
    quads = []
    epsilons = np.linspace(0, 0.2, 100)
    for epsilon in epsilons:
        lin = np.linalg.norm(f(x+epsilon*d)-f(x))
        quad = np.linalg.norm(f(x+epsilon*d)-f(x)-j(x, epsilon*d))
        lins.append(lin)
        quads.append(quad*4)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Jacobian test')
    plt.plot(epsilons, lins)
    plt.plot(epsilons, quads)
    plt.legend(['$|f(x)+\\epsilon\\cdot d)-f(x)|$',
                '$|f(x+\\epsilon\\cdot d)-f(x)-JacMV(x,\\epsilon d)|$'])
    plt.savefig('jacobian_test.png')


"""
x = np.array([[1, 2, 3], [4, 5, 6]])
i = np.identity(3)
t1 = np.tensordot(x[0], i, axes=0).reshape(9, 3).T
t2 = np.tensordot(x[1], i, axes=0).reshape(9, 3).T
t = np.concatenate((t1, t2))
print(t1)
print(t1.shape)
exit()
"""

input_dim = 3
output_dim = 2
x = np.random.random((3, input_dim)).T
x = np.array([[2, 5, 1], [1, 1, 1], [2, 4, 2]]).T
y_arr = []
for i in range(100):
    if i % 2 == 0:
        y_arr.append(np.array([1, 0]))
    else:
        y_arr.append(np.array([0, 1]))
y = np.array(y_arr).T
n = Network()
n.add_layer(Layer(input_dim, output_dim, TANH_ACTIVATION))
n.add_layer(Layer(output_dim, 2, None, softmax_layer=True))
curr_batch = [0]
#batch_x = np.array([x.T[ind] for ind in curr_batch]).T
#batch_y = np.array([y.T[ind] for ind in curr_batch]).T
l = n.get_layer(0)


x = np.array([[2, 5, 1]])

def f(inp):
    l.set_weights(inp, update_bias=True)
    x1 = x[0]
    mf_net = l.forward_pass(x1)
    #print('mf net', mf_net)
    w = inp[:6].reshape(2, 3)
    b = inp[6:]
    my_forward = np.atleast_2d(np.tanh(np.dot(w, x1.T)+b).flatten()).T
    print('mf', my_forward)
    diff = my_forward-mf_net
    if np.any(np.abs(diff) > 0.00001):
        raise BaseException("aaa")
    return mf_net
    return my_forward
    return l.forward_pass(x1)
    return np.tanh(np.dot(w, x1.T)+b).flatten()


def jacob(inp, v):
    l.set_weights(inp, update_bias=True)
    inp = np.concatenate((inp[:6].reshape(2, 3).T.flatten(), inp[6:]))
    x1 = x[0]
    l.forward_pass(x1)
    l.calc_jacobian()
    j = l.get_jacobian()
    ans = np.dot(j, v)
    ans = np.atleast_2d(np.dot(j, v)).T
    #print(np.atleast_2d(np.dot(j, v)).T)
    return ans
    # print(inp)
    x1 = x[0]
    x2 = x[0]
    x3 = x[0]
    w = inp[:6].reshape(3, 2).T.flatten()
    b = inp[6:]
    w1 = w[:3]
    w2 = w[3:]
    wm = inp[:6].reshape(2, 3)
    t1 = np.dot(x1, w1)+b[0]
    t2 = np.dot(x1, w2)+b[1]
    t3 = np.dot(x2, w1)+b[0]
    t4 = np.dot(x2, w2)+b[1]
    t5 = np.dot(x3, w1)+b[0]
    t6 = np.dot(x3, w2)+b[1]
    t1 = 1-np.tanh(t1)**2
    t2 = 1-np.tanh(t2)**2
    t3 = 1-np.tanh(t3)**2
    t4 = 1-np.tanh(t4)**2
    t5 = 1-np.tanh(t5)**2
    t6 = 1-np.tanh(t6)**2
    mat = np.array([[x1[0]*t1, 0, x1[1]*t1, 0, x1[2]*t1, 0, t1, 0],
                    [0, x1[0]*t2, 0, x1[1]*t2, 0, x1[2]*t2, 0, t2]])
    mat2 = np.array([[x1[0]*t1, x1[1]*t1, x1[2]*t1, 0, 0, 0, t1, 0],
                    [0, 0, 0, x1[0]*t2, x1[1]*t2, x1[2]*t2, 0, t2],
                    [x2[0]*t3, x2[1]*t3, x2[2]*t3, 0, 0, 0, t3, 0],
                    [0, 0, 0, x2[0]*t4, x2[1]*t4, x2[2]*t4, 0, t4],
                    [x3[0] * t5, x3[1] * t5, x3[2] * t5, 0, 0, 0, t5, 0],
                    [0, 0, 0, x3[0] * t6, x3[1] * t6, x3[2] * t6, 0, t6]])
    print('a', mat)
    print('a', inp)
    diff = np.dot(j, v)-np.dot(mat, v)
    #exit()
    return np.dot(j, v)

def f_network(inp):
    n.set_weights(inp)
    return n.calc_error(batch_x, batch_y)


def grad_network(inp):
    n.set_weights(inp)
    return n.get_grad(batch_x, batch_y)


dim = input_dim*output_dim+output_dim
#dim = 8
jacobian_test(f, jacob, dim)
#gradient_test(f_network, grad_network, (n.get_param_num(), ))
exit()
num_of_labels = y.shape[0]
wb = np.concatenate((w.flatten(), b.T), axis=None)
w, obj_train = stochastic_gradient_descent(len(x.T), num_of_labels,
                                           obj, grad, wb.shape, batch_size=100)
w_range = x.shape[0] * num_of_labels
curr_w = wb[:w_range].reshape(num_of_labels, x.shape[0])
w = curr_w
plt.plot(range(len(obj_train)), obj_train)
plt.show()
