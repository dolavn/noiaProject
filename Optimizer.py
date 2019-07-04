import numpy as np

MAX_ITER = 40

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
    print(batches)
    return batches


def stochastic_gradient_descent(data_size, num_of_labels, objective, gradient, input_shape,
                                batch_size=1):
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
    for _ in r:
        curr_batch = batches[curr_batch_ind]
        curr_batch_ind = (curr_batch_ind+1) % len(batches)
        g = gradient(w, curr_batch)
        obj_train.append(objective(w, range(data_size)))
        alpha = get_step_size(w, lambda x: objective(x, curr_batch), g, -g)
        #alpha = 0.01
        if alpha == 0:
            break
        w = w - alpha*g
        if np.linalg.norm(g)/np.linalg.norm(grad1) <= EPSILON:
            break
    #obj_train = [np.abs(obj - objective(w, range(data_size))) for obj in obj_train]
    return w, obj_train
