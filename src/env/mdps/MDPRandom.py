import numpy as np


class GARNET:

    def __init__(self, num_states, num_actions, branching_factor, reward_sparsity, seed):
        self.num_states = num_states
        self.num_actions = num_actions
        self.branching_factor = branching_factor
        self.reward_sparsity = reward_sparsity
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))
        self.create_new_MDP()

    def create_new_MDP(self):
        self.P = np.zeros((self.num_actions, self.num_states, self.num_states))
        self.R = self.random.randn(self.num_states, self.num_actions)
        self.R *= self.random.choice(a=2, size=(self.num_states, self.num_actions), p=[1-self.reward_sparsity, self.reward_sparsity])
        for a in range(self.num_actions):
            for s in range(self.num_states):
                prob = self.random.uniform(size=(self.branching_factor,))
                prob = prob / np.sum(prob)
                indices = self.random.choice(self.num_states,size=(self.branching_factor,), replace=False)
                self.P[a, s, indices] = prob

    def __str__(self):
        return str(self.P) + "\n" + str(self.R)





