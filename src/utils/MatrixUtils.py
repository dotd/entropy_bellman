import numpy as np


def get_transition_matrix(random_numpy_generator, size_y, size_x):
    """
    Get a random transition matrix
    :param random_numpy_generator: random generator (numpy based) for making the transition matrix (not necessarily square)
    :param size_y: lines on matrix
    :param size_x: columns of matrix
    :return:
    """
    p = random_numpy_generator.rand(size_y, size_x)
    s = np.sum(p, axis=1)
    s = np.expand_dims(s, 1)
    p = p / s  # normalizing the matrix
    return p

