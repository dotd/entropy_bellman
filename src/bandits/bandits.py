import numpy as np


class Bandits:
    '''
    This class generates several bandits with different probabilities.
    The main functions are
    1) Create the settings.
    2) Get next sample according to the index.
    '''

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

