import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import Network

MAX_ITER = 50

ALPHA_0 = 1
BETA = 0.5
C = 10e-6
EPSILON = 10e-4
CALC_ERROR_EVERY = 120


def get_step_size(x, obj, grad, dir):
    """
    Chooses a value for the step size using Armijo's rule
    :param x: A vector x
    :param obj:  The objective function
    :param grad:  The gradient of the objective function
    :param dir: The search direction
    :return: The value of the step size
    """
    alpha = ALPHA_0
    #dir = dir/np.linalg.norm(dir)
    for j in range(MAX_ITER):
        phi = obj(x+alpha*dir)
        m = np.dot(grad, dir)
        #print(phi)
        #print(obj(x)+C*alpha*np.dot(grad, dir))
        if phi <= obj(x)+C*alpha*np.dot(grad, dir):
            print(alpha)
            return alpha
        else:
            alpha = BETA*alpha
    return 0


def create_batches(data_size, batch_size):
    num_of_batches = int(data_size/batch_size) + (1 if data_size % batch_size != 0 else 0)
    curr_ind = 0
    batches = []
    indices = np.random.permutation(range(data_size))
    for i in range(num_of_batches):
        last_ind = curr_ind
        curr_ind = min(curr_ind+batch_size, data_size)
        batches.append(indices[last_ind: curr_ind])
    #print(batches)
    return batches

TOTAL_DOTS = 50

def print_percentage(curr_ind, max_ind):
    percantage = 100*(curr_ind/max_ind)
    num_pos = int(TOTAL_DOTS*(curr_ind/max_ind))
    num_neg = TOTAL_DOTS-num_pos
    output = '[' + (num_pos-1)*'=' + '>' + '.'*num_neg + '][{}%]'.format(percantage)
    print(output)


def stochastic_gradient_descent(network, x, y, batch_size=100, epochs=1, learning_rate=0.1,
                                x_validation=None, y_validation=None, extra_plot_func=None,
                                extra_text=None, decay_factor=1):
    n, m = x.shape
    obj_train = []
    obj_valid = []
    obj = network.calc_error(x, y)
    if x_validation is not None:
        obj_v = network.calc_error(x_validation, y_validation)
        obj_valid.append(obj_v)
    obj_train.append(obj)
    curr_learning_rate = learning_rate
    for epoch in range(epochs):
        print('Epoch {} objective training {}, objective validation {}'.format(epoch+1,
                                                                               obj_train[-1],
                                                                               obj_valid[-1]))
        batches = create_batches(m, batch_size)
        for curr_batch_ind, batch in enumerate(batches):
            print_percentage(curr_batch_ind, len(batches))
            curr_x = np.array([x.T[i] for i in batch]).T
            curr_y = np.array([y.T[i] for i in batch]).T
            if curr_batch_ind % CALC_ERROR_EVERY == 0:
                obj = network.calc_error(x, y)
                if x_validation is not None:
                    obj_v = network.calc_error(x_validation, y_validation)
                    obj_valid.append(obj_v)
                obj_train.append(obj)
            if curr_batch_ind % (len(batches)/2) == 0:
                if extra_plot_func:
                    extra_plot_func(network, 2*epoch + (1 if curr_batch_ind != 0 else 0))
            grad = network.get_grad(curr_x, curr_y)
            grad = -grad*curr_learning_rate
            network.inc_weights(grad)
            curr_learning_rate = curr_learning_rate * decay_factor
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    epoch_string = 'epochs' if epochs > 1 else 'epoch'
    extra_txt_fname = ''
    if extra_text:
        for ind, text in enumerate(extra_text):
            extra_txt_fname = extra_txt_fname + '_' + text
            ax1.text(0.1, 0.6-0.05*ind, text,
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=ax1.transAxes)
    ax1.set_title('Objective function {} {}, batch_size {} ,'
                  ' learning_rate {}'.format(epochs,
                                             epoch_string,
                                             batch_size,
                                             learning_rate))
    plt.plot(range(len(obj_train)), obj_train, label='training')
    if len(obj_valid) > 0:
        plt.plot(range(len(obj_valid)), obj_valid, label='validation')
    ax1.legend()
    plt.savefig('objective_func_e{}_b{}_lr{}{}.png'.format(epochs, batch_size, learning_rate,
                                                           extra_txt_fname))
    return network, obj_train

