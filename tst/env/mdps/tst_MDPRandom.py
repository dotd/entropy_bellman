from src.env.mdps.MDPRandom import GARNET


def tst_GARNET():
    X = 5
    A = 3
    B = 3
    reward_sparsity = 0.5
    mdp = GARNET(X, A, B, reward_sparsity, seed=0)
    print(mdp)


if __name__ == "__main__":
    tst_GARNET()