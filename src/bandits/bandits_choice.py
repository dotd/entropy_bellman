import numpy as np
from src.utils.MatrixUtils import get_transition_matrix


class Bandits:

    def __init__(self, seed, vec_size, num_bandits):
        self.seed = seed
        self.vec_size = vec_size
        self.num_bandits = num_bandits
        self.bandits_prob = np.zeros(shape=[num_bandits, vec_size])
        self.random = list()

        # First we randomize the vectors
        for i in range(self.num_bandits):
            self.random.append(np.random.RandomState(seed + i))
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
        self.bandits_samples = np.zeros(shape=(num_bandits, vec_size))
        self.prob_hat = get_transition_matrix(self.random, num_bandits, vec_size)
        self.prob_hat_prev = get_transition_matrix(self.random, num_bandits, vec_size)
        self.prob_hat_diff_norm = np.sum(np.abs(self.prob_hat_prev - self.prob_hat), axis=1)
        self.error_L1_from_true = np.zeros(shape=(self.num_bandits,))
        self.total_error = None

    def update_stats_L1(self, true_stats):
        for index in range(self.num_bandits):
            self.error_L1_from_true[index] = np.sum(np.abs(true_stats[index] - self.prob_hat[index])) / self.vec_size
        self.total_error = np.sum(self.error_L1_from_true)

    def update_stats(self, bandit_index, sample, true_stats=None):
        self.bandits_samples[bandit_index][sample] += 1
        self.prob_hat_prev[bandit_index] = self.prob_hat[bandit_index]
        self.prob_hat[bandit_index] = self.bandits_samples[bandit_index] / np.sum(self.bandits_samples[bandit_index])
        self.prob_hat_diff_norm[bandit_index] = np.sum(np.abs(self.prob_hat_prev[bandit_index] - self.prob_hat[bandit_index]))
        if true_stats is not None:
            self.update_stats_L1(true_stats)

    def get_next_bandit_random(self):
        bandit_index = self.random.choice(self.num_bandits)
        return bandit_index

    def get_max_random_on_vector(self, vec):
        maximal_value = np.amax(vec)
        maximal_indices = []
        for i, value in enumerate(vec):
            if value == maximal_value:
                maximal_indices.append(i)
        vec_maximal = np.array(maximal_indices)
        vec_maximal = self.random.permutation(vec_maximal)
        return vec_maximal[0]

    def get_next_bandit_oracle(self):
        idx = self.get_max_random_on_vector(self.error_L1_from_true)
        return idx

    def get_next_bandit_by_tail_convergence(self):
        idx = self.get_max_random_on_vector(self.prob_hat_diff_norm)
        # print(self.prob_hat_diff_norm)
        return idx

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


