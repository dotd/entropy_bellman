import matplotlib.pyplot as plt
import numpy as np

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
        # print(i)
        bandit_index = estimator.get_next_bandit_oracle()
        sample = bandits.get_sample(bandit_index=bandit_index)
        estimator.update_stats(bandit_index=bandit_index, sample=sample, true_stats=bandits.bandits_prob)
        # print(estimator.to_string())

        err.append(estimator.total_error)
    return err


def run_tail_strategy(seed, vec_size, num_bandits, num_steps):
    # Random Strategy
    bandits = Bandits(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    estimator = BanditsEstimator(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    err = []
    bandit_aggregator = []
    for i in range(num_steps):
        # print(i)
        bandit_index = estimator.get_next_bandit_by_tail_convergence()
        bandit_aggregator.append(bandit_index)
        sample = bandits.get_sample(bandit_index=bandit_index)
        estimator.update_stats(bandit_index=bandit_index, sample=sample, true_stats=bandits.bandits_prob)
        # print(estimator.to_string())
        print(estimator.prob_hat_diff_norm)
        print("")

        err.append(estimator.total_error)
    return err, bandit_aggregator


def compute_percentage_above(sig1, sig2):
    total = len(sig1)
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)
    den = np.sum((np.sign(sig1-sig2) + 1) / 2)
    rate = den / total
    return rate


def experiment_random_vs_oracle_single(seed,
                                       vec_size,
                                       num_bandits,
                                       num_steps,
                                       figure_flag,
                                       block_flag,
                                       figure_tail_flag=True):

    err_random = run_random_strategy(seed=seed, vec_size=vec_size, num_bandits=num_bandits, num_steps=num_steps)
    err_oracle = run_oracle_strategy(seed=seed, vec_size=vec_size, num_bandits=num_bandits, num_steps=num_steps)
    err_tail, bandit_aggregator = run_tail_strategy(seed=seed, vec_size=vec_size, num_bandits=num_bandits, num_steps=num_steps)

    # Oracle - knowing where is the problem

    if figure_flag:
        plt.figure()
        plt.plot(err_random, label="random")
        plt.plot(err_oracle, label="oracle")
        plt.plot(err_tail, label="tail")
        plt.ylabel('error')
        plt.legend()
        plt.show(block=block_flag)

    if figure_tail_flag:
        plt.figure()
        plt.plot(bandit_aggregator, '.')
        plt.show()

    diff_oracle_minus_random = np.array(err_oracle) - np.array(err_random)
    rate = compute_percentage_above(err_oracle, err_random)
    return rate, diff_oracle_minus_random


def experiment_random_vs_oracle():
    vec_size = 8
    num_bandits = 5
    num_steps = int(200)
    num_experiments = 10
    figure_flag = True
    block_flag = True
    rates = []
    rates_aggregate = []
    for seed in range(num_experiments):
        print(f"experiment {seed}")
        rate, diff_oracle_minus_random = experiment_random_vs_oracle_single(seed=seed,
                                                                            vec_size=vec_size,
                                                                            num_bandits=num_bandits,
                                                                            num_steps=num_steps,
                                                                            figure_flag=figure_flag,
                                                                            block_flag=block_flag)
        rates_aggregate.append(diff_oracle_minus_random.tolist())
        rates.append(rate)
        rate_mean = np.mean(rates)
        print(f"rate_mean={rate_mean}")

    rates_aggregate = np.array(rates_aggregate).T
    rate_mean2 = np.mean((np.sign(rates_aggregate) + 1) / 2)
    print(f"rate_mean2_sign={rate_mean2}")
    plt.figure()
    plt.plot(rates_aggregate)
    plt.show(block=True)


if __name__ == "__main__":
    experiment_random_vs_oracle()
    # input("Press Enter to continue...")
