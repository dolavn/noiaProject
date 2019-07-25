import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import Network, Layer, TANH_ACTIVATION, softmax_obj, sm_activation_gx


def gradient_test(f, g, dim, title='Gradient test', file_name='gradient_test.png'):
    """
    Performs the gradient test on f
    :param f: A function
    :param g: The gradient of f
    :param dim: The dimensions of the input to f
    :return: None
    """
    x = np.random.rand(*dim)
    if dim == 2:
        x = np.array([1, 2])
    d = np.random.rand(*dim)
    d = d/np.linalg.norm(d)
    lins = []
    quads = []
    epsilons = np.linspace(0, 0.5, 200)
    for epsilon in epsilons:
        assert (x.shape == d.shape)
        lin = np.abs(f(x+epsilon*d)-f(x))
        d_flat = d.flatten()
        g_flat = g(x).flatten()
        dot = np.dot(d_flat, g_flat)
        assert(f(x+epsilon*d).shape == f(x).shape)
        assert (f(x).shape == dot.shape)
        quad = np.abs(f(x+epsilon*d)-f(x)-epsilon*dot)
        lins.append(lin)
        quads.append(quad*5)
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    plt.plot(epsilons, lins)
    plt.plot(epsilons, quads)
    plt.legend(['$|f(x)+\\epsilon\\cdot d)-f(x)|$',
                '$|f(x+\\epsilon\\cdot d)-f(x)-\\epsilon d^{\\top}\\cdot grad(x)|$'])
    plt.savefig(file_name)


def jacobian_test(f, j, dim, title='Jacobian Test', file_name='jacobian_test.png'):
    """
    Performs the jacobian test on f
    :param f: A function f
    :param j: A method that calculates M*v where v is a vector and M is the Jacobian
    :param dim: The dimensions of the input to f
    :return: None
    """
    x = np.random.rand(dim)
    #x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.1])
    d = np.random.rand(dim)
    d = d/np.linalg.norm(d)
    lins = []
    quads = []
    epsilons = np.linspace(0, 0.2, 100)
    for epsilon in epsilons:
        assert(f(x+epsilon*d).shape == j(x, epsilon*d).shape)
        assert (f(x + epsilon * d).shape == f(x).shape)
        lin = np.linalg.norm(f(x+epsilon*d)-f(x))
        quad = np.linalg.norm(f(x+epsilon*d)-f(x)-j(x, epsilon*d))
        lins.append(lin)
        quads.append(quad*4)
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    plt.plot(epsilons, lins)
    plt.plot(epsilons, quads)
    plt.legend(['$|f(x)+\\epsilon\\cdot d)-f(x)|$',
                '$|f(x+\\epsilon\\cdot d)-f(x)-JacMV(x,\\epsilon d)|$'])
    plt.savefig(file_name)


FIRST_LAYER_INPUT = 10
SECOND_LAYER_INPUT = 2
LABELS_NUM = 2
NUM_OF_SAMPLES = 5
x = np.random.random((NUM_OF_SAMPLES, FIRST_LAYER_INPUT)).T
y_arr = []
for i in range(NUM_OF_SAMPLES):
    if i % 2 == 0:
        y_arr.append(np.array([1, 0]))
    else:
        y_arr.append(np.array([0, 1]))
y = np.array(y_arr).T
n = Network()
n.add_layer(Layer(FIRST_LAYER_INPUT, SECOND_LAYER_INPUT, TANH_ACTIVATION))
n.add_layer(Layer(SECOND_LAYER_INPUT, LABELS_NUM, None, softmax_layer=True))
curr_batch = np.random.permutation(range(NUM_OF_SAMPLES))
batch_x = np.array([x.T[ind] for ind in curr_batch]).T
batch_y = np.array([y.T[ind] for ind in curr_batch]).T
l = n.get_layer(0)


# Softmax test functions

l_sm = Layer(FIRST_LAYER_INPUT, LABELS_NUM, None, softmax_layer=True)

W_EXAMPLE = np.array([[2, 0], [0, 1]]).T
B_EXAMPLE = (np.array([1, 1]))
C1_EXAMPLE = np.atleast_2d(np.array([0])).T
C2_EXAMPLE = np.atleast_2d(np.array([1])).T
Y_EXAMPLE = np.atleast_2d(np.array([0, 1])).T


def f_softmax_example(inp):
    x1 = inp[0]
    x2 = inp[1]
    a1 = np.log(np.exp(2*x1+1)+np.exp(x2+1))-x2-1
    a2 = softmax_obj(np.atleast_2d(inp).T, Y_EXAMPLE, W_EXAMPLE.T, B_EXAMPLE)
    assert a1 - a2 < 0.0001
    return a2


def g_softmax_example(inp):
    x1 = inp[0]
    x2 = inp[1]
    a1 = np.atleast_2d(np.array([2*np.exp(2*x1+1)/(np.exp(2*x1+1)+np.exp(x2+1)),
                       np.exp(x2+1)/(np.exp(2*x1+1)+np.exp(x2+1))-1])).T
    a2 = sm_activation_gx(np.atleast_2d(inp).T, Y_EXAMPLE, W_EXAMPLE, B_EXAMPLE)
    #print(a1)
    #print(a2)
    assert np.all(a1 - a2) < 0.0001
    return a2


def f_softmax(inp):
    l_sm.set_weights(inp)
    l_sm.forward_pass(batch_x)
    return l_sm.calc_softmax_obj(batch_y)


def g_softmax(inp):
    l_sm.set_weights(inp)
    l_sm.forward_pass(batch_x)
    l_sm.calc_softmax_grad(batch_y)
    return l_sm.get_theta_grad()

# Softmax gradient according to data test functions


def f_softmax_data(inp):
    l_sm.forward_pass(inp.reshape(FIRST_LAYER_INPUT, NUM_OF_SAMPLES))
    return l_sm.calc_softmax_obj(batch_y)


def g_softmax_data(inp):
    l_sm.forward_pass(inp.reshape(FIRST_LAYER_INPUT, NUM_OF_SAMPLES))
    l_sm.calc_softmax_grad(batch_y)
    delta = l_sm.get_delta()
    return delta


# Jacobian test functions


def f_jacob(inp):
    l.set_weights(inp)
    mf_net = np.atleast_2d(l.forward_pass(batch_x).T.flatten()).T
    return mf_net


def jacob(inp, v):
    l.set_weights(inp)
    l.forward_pass(batch_x)
    l.calc_jacobian()
    j = l.get_jacobian()
    return np.atleast_2d(np.dot(j, v)).T


def f_jacob_data(inp):
    #l.set_weights(inp)
    inp = inp.reshape(*batch_x.shape)
    mf_net = np.atleast_2d(l.forward_pass(inp).T.flatten()).T
    return mf_net


def jacob_data(inp, v):
    #l.set_weights(inp)
    inp = inp.reshape(*batch_x.shape)
    l.forward_pass(inp)
    l.calc_jacobian()
    j = l.get_jacobian_data()
    return np.atleast_2d(np.dot(j, v)).T



# Gradient network test functions

def f_network(inp):
    n.set_weights(inp)
    return n.calc_error(batch_x, batch_y)


def grad_network(inp):
    n.set_weights(inp)
    return n.get_grad(batch_x, batch_y)



print('Running Gradient test on softmax')
gradient_test(f_softmax_example, g_softmax_example, (2, ),
              title='Softmax gradient test', file_name='softmax_example.png')
print('Running Gradient test on softmax')
gradient_test(f_softmax, g_softmax, (FIRST_LAYER_INPUT*LABELS_NUM+LABELS_NUM, ),
              title='Softmax gradient test', file_name='gradient_test_softmax.png')
print('Running Gradient test on softmax according to data')
gradient_test(f_softmax_data, g_softmax_data, (FIRST_LAYER_INPUT*NUM_OF_SAMPLES, ),
              title='Softmax gradient test - data', file_name='gradient_test_softmax_data.png')
print('Running Jacobian test')
jacobian_test(f_jacob, jacob, FIRST_LAYER_INPUT*SECOND_LAYER_INPUT+SECOND_LAYER_INPUT,
              title='Jacobian test',
              file_name='jacobian_test.png')
print('Running Jacobian test data')
jacobian_test(f_jacob_data, jacob_data, batch_x.shape[0]*batch_x.shape[1],
              title='Jacobian test',
              file_name='jacobian_test_data.png')
#exit()
print('Running Gradient test on network')
gradient_test(f_network, grad_network, (n.get_param_num(), ),
              title='Network gradient test', file_name='gradient_test_network.png')
