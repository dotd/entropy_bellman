import numpy as np
from src.utils.MatrixUtils import get_transition_matrix


class BanditsEstimator:
    """
    BanditsEstimator estimates the stats of the bandits.
    """

    def __init__(self,
                 seed,  # Seed is mainly needed for breaking evens randomly.
                 vec_size,
                 num_bandits):
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.vec_size = vec_size
        self.num_bandits = num_bandits

        # Accumulates the samples for each bandit
        self.bandits_samples = np.zeros(shape=(num_bandits, vec_size))

        # The estimation of the probability
        self.prob_hat = get_transition_matrix(self.random, num_bandits, vec_size)

        # Remember the previous estimation
        self.prob_hat_prev = get_transition_matrix(self.random, num_bandits, vec_size)

        # The L1 difference between two consecutive estimations.
        self.prob_hat_diff_norm = np.sum(np.abs(self.prob_hat_prev - self.prob_hat), axis=1)

        # The error from the true distribution
        self.error_L1_from_true = np.zeros(shape=(self.num_bandits,))

        self.total_error = None

    def update_stats(self, bandit_index, sample, true_stats=None):
        # this is counters with all the stats
        self.bandits_samples[bandit_index][sample] += 1

        # Remember the previous probability
        self.prob_hat_prev[bandit_index] = self.prob_hat[bandit_index]

        # Update the new probability
        self.prob_hat[bandit_index] = self.bandits_samples[bandit_index] / np.sum(self.bandits_samples[bandit_index])

        # How the probability has changed?
        self.prob_hat_diff_norm[bandit_index] = np.sum(np.abs(self.prob_hat_prev[bandit_index] - self.prob_hat[bandit_index]))

        if true_stats is not None:
            self.__update_stats_l1(true_stats)

    def get_next_bandit_oracle(self):
        idx = self.__get_max_random_on_vector(self.error_L1_from_true)
        return idx

    def get_next_bandit_by_tail_convergence(self):
        idx = self.__get_max_random_on_vector(self.prob_hat_diff_norm)
        # print(self.prob_hat_diff_norm)
        return idx

    # --- Private Methods ---
    def __update_stats_l1(self, true_stats):
        for index in range(self.num_bandits):
            self.error_L1_from_true[index] = np.sum(np.abs(true_stats[index] - self.prob_hat[index])) / self.vec_size
        self.total_error = np.sum(self.error_L1_from_true)

    def __get_next_bandit_random(self):
        bandit_index = self.random.choice(self.num_bandits)
        return bandit_index

    def __get_max_random_on_vector(self, vec):
        maximal_value = np.amax(vec)
        maximal_indices = []
        for i, value in enumerate(vec):
            if value == maximal_value:
                maximal_indices.append(i)
        vec_maximal = np.array(maximal_indices)
        vec_maximal = self.random.permutation(vec_maximal)
        return vec_maximal[0]

    def __str__(self):
        s = list()
        s.append(f"seed={self.seed}")
        s.append(f"vec_size={self.vec_size}")
        s.append(f"num_bandits={self.num_bandits}")
        for i in self.bandits_samples:
            s.append(f"{i}=>{self.bandits_samples[i]}")
            s.append(f"{i}=>{self.prob_hat[i]}")
        s.append(f"error={self.error_L1_from_true}")
        return "\n".join(s)


