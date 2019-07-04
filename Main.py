import numpy as np
import matplotlib.pyplot as plt


def create_c_vecs(y):
    ans = []
    for val in range(max(y)+1):
        ans.append([1 if elem == val else 0 for elem in y])
    return np.array(ans)


def softmax_obj(x, y, w, b):
    c = create_c_vecs(y)
    max_label = max(y)+1
    sum_all = sum([np.exp(np.dot(x.T, w[i])+b[i]) for i in range(max_label)])
    val = -sum([np.dot(c[i].T, np.log(np.exp(np.dot(x.T, w[i])+b[i])/sum_all)) for i in range(max_label)])/len(x)
    return val


def softmax_grad(x, y, w, b):
    c = create_c_vecs(y)
    max_label = max(y)+1
    sum_all = sum([np.exp(np.dot(x.T, w[i])+b[i]) for i in range(max_label)])
    cis = [(np.exp(np.dot(x.T, w[i])+b[i])/sum_all)-c[i] for i in range(max_label)]
    cis = [1/len(x)*np.dot(x, elem) for elem in cis]
    cis = [elem.reshape(1, -1) for elem in cis]
    ans = np.concatenate(cis, axis=0)
    bs = [np.dot(x.T, w[i]) + b[i] for i in range(max_label)]
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
        quads.append(quad)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Gradient test')
    plt.plot(epsilons, lins)
    plt.plot(epsilons, quads)
    plt.legend(['$|f(x)+\\epsilon\\cdot d)-f(x)|$',
                '$|f(x+\\epsilon\\cdot d)-f(x)-\\epsilon d^{\\top}\\cdot grad(x)|$'])
    plt.savefig('gradient_test.png')


x = np.array([[1, 2], [2, 2], [1, 0], [1, 1], [0, 2]]).T
y = np.array([2, 3, 0, 1, 1])
w = np.array([[0.2, 1], [0.1, 0], [1, 0], [0.1, 0]])
b = np.array([1, 1, 1, 1])
print(x)
print(y)
v = softmax_obj(x, y, w, b)
print(v)
vs = [v]
#gradient_test(lambda a: softmax_obj(x, y, a, b),
#              lambda a: softmax_grad(x, y, a, b),
#              w.shape)
for _ in range(100):
    g, gb = softmax_grad(x, y, w, b)
    w = w - 0.1*g
    b = b - 0.1*gb
    vs.append(softmax_obj(x, y, w, b))
v = softmax_obj(x, y, w, b)
print(w, b)
print(v)
plt.plot(range(len(vs)), vs)
plt.show()
