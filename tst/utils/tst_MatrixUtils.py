import numpy as np
from src.utils.MatrixUtils import get_transition_matrix

if __name__ == "__main__":
    random = np.random.RandomState(0)
    pp = get_transition_matrix(random, 4, 5)
    print(pp)
    print(np.sum(pp, axis=1, keepdims=True))
