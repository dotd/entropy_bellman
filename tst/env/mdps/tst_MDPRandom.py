import numpy as np
from src.env.mdps.MDPRandom import GARNET


def tst_GARNET(seed=1):
    X = 5
    A = 3
    B = 3
    random = np.random.RandomState(seed)
    reward_sparsity = 0.5
    contrast = 1
    mdp = GARNET(X, A, B, reward_sparsity, seed=0, contrast=contrast)
    print(f"MDP is:\n{mdp}\n------")
    num_trajectories = 2
    trajectory_length = 10
    trajectories = list()
    for e in range(num_trajectories):
        state = mdp.reset()
        trajectory = list()
        for t in range(trajectory_length):
            action = random.choice(mdp.num_actions)
            state_next, reward, done, info = mdp.step(action)
            trajectory.append([state, action, reward, state_next])
            state = state_next
        trajectories.append(trajectory)

    print(np.array(trajectories))


if __name__ == "__main__":
    tst_GARNET()
