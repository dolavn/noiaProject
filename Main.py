import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io
from Optimizer import stochastic_gradient_descent
from NeuralNetwork import Network, TANH_ACTIVATION, RELU_ACTIVATION, Layer
from itertools import product

mat = scipy.io.loadmat('SwissRollData.mat')
labels = mat['Ct']
training = mat['Yt']
x = training
y = labels
n = Network()
n.add_layer(Layer(2, 10, TANH_ACTIVATION))
n.add_layer(Layer(10, 10, TANH_ACTIVATION))
n.add_layer(Layer(10, 2, None, softmax_layer=True))
n, obj = stochastic_gradient_descent(n, x, y, batch_size=100, epochs=6)
fig1, ax1 = plt.subplots()
x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-1.5, 1.5, 100)
image = np.zeros((100, 100))
image2 = np.zeros((100, 100))
for (i1, x_i), (i2, y_i) in product(enumerate(x_range),
                                    enumerate(y_range)):
    vec = np.atleast_2d(np.array([x_i, y_i]))
    label = n.forward_pass(vec.T)[0]
    image[i1][i2] = label[0]
plt.imshow(image, extent=[-1.5, 1.5, -1.5, 1.5])
coord_x_pos = [x.T[ind][0] for ind in range(y.shape[1]) if y.T[ind][0] == 1]
coord_y_pos = [x.T[ind][1] for ind in range(y.shape[1]) if y.T[ind][0] == 1]
coord_x_neg = [x.T[ind][0] for ind in range(y.shape[1]) if y.T[ind][0] == 0]
coord_y_neg = [x.T[ind][1] for ind in range(y.shape[1]) if y.T[ind][0] == 0]
plt.scatter(coord_y_pos, coord_x_pos, alpha=0.2)
plt.scatter(coord_y_neg, coord_x_neg, alpha=0.2)
plt.show()
exit()
