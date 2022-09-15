import numpy as np


def get_transition_matrix(random, size_y, size_x):
    p = random.rand(size_y, size_x)
    s = np.sum(p, axis=1)
    s = np.expand_dims(s, 1)
    p = p / s
    return p


if __name__ == "__main__":
    random = np.random.RandomState(0)
    pp = get_transition_matrix(random, 4, 5)
    print(pp)
    print(np.sum(pp, axis=1, keepdims=True))
