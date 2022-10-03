import numpy as np
import scipy as sp


def divergence_deterministic_vs_random(p1, p2):
    pass


def create_random_distribution(target_entropy, rnd, tolerance, initial_distribution):
    np.set_printoptions(precision=3, suppress=True)
    distribution = initial_distribution
    h = sp.stats.entropy(distribution)
    n = 1
    print(f"n={n}\ndistribution={distribution}\nh={h}\ntarget_entropy={target_entropy}\n")
    while abs(h-target_entropy) > tolerance:
        if h < target_entropy:
            distribution = np.power(distribution, 1 / (1 + 0.5 / (n + 1)))
        else:
            distribution = np.power(distribution, 1 / (1 - 0.5 / (n + 1)))
        distribution = distribution / np.sum(distribution)
        h = sp.stats.entropy(distribution)
        n += 1
        print(f"n={n}\ndistribution={distribution}\nh={h}\ntarget_entropy={target_entropy}\n")
    return distribution


def get_random_distribution(dimension, rnd):
    vec = rnd.uniform(size=(dimension,))
    vec = vec / np.sum(vec)
    return vec


def basic_experiment():
    dimension = 10
    target_entropy = 0.8
    tolerance = 0.01
    rnd = np.random.RandomState(1)
    initial_distribution = get_random_distribution(dimension, rnd)
    create_random_distribution(target_entropy=target_entropy,
                               rnd=rnd,
                               tolerance=tolerance,
                               initial_distribution=initial_distribution)


if __name__ == "__main__":
    basic_experiment()
