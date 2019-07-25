import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import Network

MAX_ITER = 50

ALPHA_0 = 1
BETA = 0.5
C = 10e-6
EPSILON = 10e-4


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
    output = '[' + (num_pos-1)*'=' + '>' + ' '*num_neg + '][{}%]'.format(percantage)
    print(output)


def stochastic_gradient_descent(network, x, y, batch_size=100, epochs=1):
    """
    Performs a general one point iterative method on a given objective function.
    :param objective: The objective function
    :param gradient: The gradient of the objective function
    :param input_shape: The shape of the input
    :return: The optimal value of w, and the history of convergence on the train dataset,
    and test dataset if given.
    """
    n, m = x.shape
    batches = create_batches(m, batch_size)
    obj_train = []
    alpha = 0.01
    for epoch in range(epochs):
        print('Epoch {}'.format(epoch+1))
        for curr_batch_ind, batch in enumerate(batches):
            print_percentage(curr_batch_ind, len(batches))
            curr_x = np.array([x.T[i] for i in batch]).T
            curr_y = np.array([y.T[i] for i in batch]).T
            obj = network.calc_error(x, y)
            obj_train.append(obj)
            grad = network.get_grad(curr_x, curr_y)
            grad = -grad*alpha
            network.inc_weights(grad)
            #alpha = get_step_size(w, lambda x: objective(x, curr_batch), g, -g)
    #obj_train = [np.abs(obj - objective(w, range(data_size))) for obj in obj_train]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Objective function')
    plt.plot(range(len(obj_train)), obj_train)
    plt.savefig('objective_func.png')
    return network, obj_train


def stochastic_gradient_descent_old(data_size, objective, gradient, input_shape,
                                    batch_size=100):
    """
    Performs a general one point iterative method on a given objective function.
    :param objective: The objective function
    :param gradient: The gradient of the objective function
    :param input_shape: The shape of the input
    :return: The optimal value of w, and the history of convergence on the train dataset,
    and test dataset if given.
    """
    w = np.random.rand(*input_shape)
    batches = create_batches(data_size, batch_size)
    curr_batch_ind = 0
    obj_train = []
    r = range(MAX_ITER)
    curr_batch = batches[curr_batch_ind]
    grad1 = gradient(w, curr_batch)
    for k in r:
        curr_batch = batches[curr_batch_ind]
        curr_batch_ind = (curr_batch_ind+1) % len(batches)
        g = gradient(w, curr_batch)
        obj_train.append(objective(w, range(data_size)))
        #alpha = get_step_size(w, lambda x: objective(x, curr_batch), g, -g)
        alpha = 0.01 if k < 100 else 0.001
        if alpha == 0:
            break
        w = w - alpha*g
        if np.linalg.norm(g)/np.linalg.norm(grad1) <= EPSILON:
            break
    #obj_train = [np.abs(obj - objective(w, range(data_size))) for obj in obj_train]
    plt.plot(range(len(obj_train)), obj_train)
    plt.show()
    return w, obj_train
