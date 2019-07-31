import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io
from Optimizer import stochastic_gradient_descent
from NeuralNetwork import Network, TANH_ACTIVATION, RELU_ACTIVATION, Layer
from itertools import product

def plot_spiral(network, ind):
    fig1, ax1 = plt.subplots()
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-1.5, 1.5, 100)
    image = np.zeros((100, 100))
    image2 = np.zeros((100, 100))
    for (i1, x_i), (i2, y_i) in product(enumerate(x_range),
                                        enumerate(y_range)):
        vec = np.atleast_2d(np.array([x_i, y_i]))
        label = network.forward_pass(vec.T)[0]
        image[len(x_range)-1-i1][i2] = label[0]
    plt.imshow(image, extent=[-1.5, 1.5, -1.5, 1.5])
    coord_x_pos = [x.T[ind][0] for ind in range(y.shape[1]) if y.T[ind][0] == 1]
    coord_y_pos = [x.T[ind][1] for ind in range(y.shape[1]) if y.T[ind][0] == 1]
    coord_x_neg = [x.T[ind][0] for ind in range(y.shape[1]) if y.T[ind][0] == 0]
    coord_y_neg = [x.T[ind][1] for ind in range(y.shape[1]) if y.T[ind][0] == 0]
    plt.scatter(coord_y_pos, coord_x_pos, alpha=0.2)
    plt.scatter(coord_y_neg, coord_x_neg, alpha=0.2)
    plt.savefig('spirals/plot_{}.png'.format(ind))

mat = scipy.io.loadmat('SwissRollData.mat')
labels = mat['Ct']
training = mat['Yt']
labels_validation = mat['Cv']
samples_validation = mat['Yv']
x = training
y = labels
x_validation = samples_validation
y_validation = labels_validation
n = Network()
n.add_layer(Layer(2, 10, TANH_ACTIVATION))
#n.add_layer(Layer(10, 10, TANH_ACTIVATION))
#n.add_layer(Layer(10, 10, TANH_ACTIVATION))
n.add_layer(Layer(10, 10, TANH_ACTIVATION))
n.add_layer(Layer(10, 2, None, softmax_layer=True))
epochs = [20]
batch_sizes = [100]
learning_rates = [0.5]
for epoch, batch_size, learning_rate in product(epochs,
                                                batch_sizes,
                                                learning_rates):
    print('epochs {} batch_size {} learning_rate {}'.format(epoch,
                                                            batch_size,
                                                            learning_rate))
    n, obj = stochastic_gradient_descent(n, x, y, batch_size=batch_size,
                                         epochs=epoch, learning_rate=learning_rate,
                                         x_validation=x_validation, y_validation=y_validation,
                                         extra_plot_func=plot_spiral,
                                         extra_text=['2hidden', 'noreg'])
    n.reset_weights()


exit()
