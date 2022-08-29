import matplotlib.pyplot as plt

from src.bandits.bandits_choice import Bandits, BanditsEstimator


def run_random_strategy(seed, vec_size, num_bandits, num_steps):
    # Random Strategy
    bandits = Bandits(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    estimator = BanditsEstimator(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    err = []
    for i in range(num_steps):
        bandit_index = estimator.get_next_bandit_random()
        sample = bandits.get_sample(bandit_index=bandit_index)
        estimator.update_stats(bandit_index=bandit_index, sample=sample, true_stats=bandits.bandits_prob)
        # print(estimator.to_string())

        err.append(estimator.total_error)
    return err


def run_oracle_strategy(seed, vec_size, num_bandits, num_steps):
    # Random Strategy
    bandits = Bandits(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    estimator = BanditsEstimator(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    err = []
    for i in range(num_steps):
        print(i)
        bandit_index = estimator.get_next_bandit_oracle()
        sample = bandits.get_sample(bandit_index=bandit_index)
        estimator.update_stats(bandit_index=bandit_index, sample=sample, true_stats=bandits.bandits_prob)
        # print(estimator.to_string())

        err.append(estimator.total_error)
    return err


def experiment_random_vs_oracle():
    seed = 1
    vec_size = 10
    num_bandits = 5
    num_steps = int(1e3)

    err_random = run_random_strategy(seed=seed, vec_size=vec_size, num_bandits=num_bandits, num_steps=num_steps)
    err_oracle = run_oracle_strategy(seed=seed, vec_size=vec_size, num_bandits=num_bandits, num_steps=num_steps)

    # Oracle - knowing where is the problem

    plt.figure()
    plt.plot(err_random, label="random")
    plt.plot(err_oracle, label="oracle")
    plt.ylabel('error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_random_vs_oracle()