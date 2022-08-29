import numpy as np
from scipy.stats import entropy


class Bandits:

    def __init__(self, seed, vec_size, num_bandits):
        self.seed = seed
        self.vec_size = vec_size
        self.num_bandits = num_bandits
        self.bandits_prob = dict()
        self.random = dict()
        self.samples = dict()

        # First we randomize the vectors
        for i in range(self.num_bandits):
            self.random[i] = np.random.RandomState(seed + i)
            self.bandits_prob[i] = self.random[i].rand(self.vec_size)
            self.bandits_prob[i] /= np.sum(self.bandits_prob[i])

    def get_sample(self, bandit_index):
        sample = self.random[bandit_index].choice(self.vec_size, p=self.bandits_prob[bandit_index])
        return sample


class BanditsEstimator:

    def __init__(self, seed, vec_size, num_bandits):
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.vec_size = vec_size
        self.num_bandits = num_bandits
        self.bandits_samples = dict()
        self.prob_hat = dict()
        self.error_L1_from_true = np.zeros(shape=(self.num_bandits,))
        self.total_error = None
        for i in range(self.num_bandits):
            self.bandits_samples[i] = np.zeros(shape=(self.vec_size,))
            self.prob_hat[i] = np.ones(shape=(self.vec_size,)) / self.vec_size

    def update_stats_L1(self, true_stats):
        for index in range(self.num_bandits):
            self.error_L1_from_true[index] = np.sum(np.abs(true_stats[index] - self.prob_hat[index])) / self.vec_size
        self.total_error = np.sum(self.error_L1_from_true)

    def update_stats(self, bandit_index, sample, true_stats=None):
        self.bandits_samples[bandit_index][sample] += 1
        self.prob_hat[bandit_index] = self.bandits_samples[bandit_index] / np.sum(self.bandits_samples[bandit_index])
        if true_stats is not None:
            self.update_stats_L1(true_stats)

    def get_next_bandit_random(self):
        bandit_index = self.random.choice(self.num_bandits)
        return bandit_index

    def get_next_bandit_oracle(self):
        maximal_value = np.amax(self.error_L1_from_true)
        maximal_indices = []
        for i, value in enumerate(self.error_L1_from_true):
            if value==maximal_value:
                maximal_indices.append(i)
        vec_maximal = np.array(maximal_indices)
        self.random.permutation(vec_maximal)
        return vec_maximal[0]


    def to_string(self):
        s = list()
        s.append(f"seed={self.seed}")
        s.append(f"vec_size={self.vec_size}")
        s.append(f"num_bandits={self.num_bandits}")
        for i in self.bandits_samples:
            s.append(f"{i}=>{self.bandits_samples[i]}")
            s.append(f"{i}=>{self.prob_hat[i]}")
        s.append(f"error={self.error_L1_from_true}")
        return "\n".join(s)


