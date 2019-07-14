import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from Optimizer import stochastic_gradient_descent
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
    epsilons = np.linspace(0, 0.5, 200)
    for epsilon in epsilons:
        lin = np.abs(f(x+epsilon*d)-f(x))
        d_flat = d.flatten()
        g_flat = g(x).flatten()
        dot = np.dot(d_flat, g_flat)
        quad = np.abs(f(x+epsilon*d)-f(x)-epsilon*dot)
        lins.append(lin)
        quads.append(quad*5)
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


num_of_labels = y.shape[0]
wb = np.concatenate((w.flatten(), b.T), axis=None)
w, obj_train = stochastic_gradient_descent(len(x.T), num_of_labels,
                                           obj, grad, wb.shape, batch_size=100)
w_range = x.shape[0] * num_of_labels
curr_w = wb[:w_range].reshape(num_of_labels, x.shape[0])
w = curr_w
plt.plot(range(len(obj_train)), obj_train)
plt.show()
