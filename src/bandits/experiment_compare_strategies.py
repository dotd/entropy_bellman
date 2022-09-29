import matplotlib.pyplot as plt
import numpy as np
from src.utils.Utils import get_time_str
from src.utils.FileUtils import create_folder_safe
from definitions import ROOT_DIR


def compute_percentage_above(sig1, sig2):
    total = len(sig1)
    sig1 = np.array(sig1)
    sig2 = np.array(sig2)
    den = np.sum((np.sign(sig1-sig2) + 1) / 2)
    rate = den / total
    return rate


def run_strategies(seed,
                   vec_size,
                   num_bandits,
                   num_steps,
                   figure_flag,
                   block_flag,
                   figure_tail_flag=True):
    """
    :param seed: seed for the run since the running is based on some randomness
    :param vec_size: dimension of each bandit distribution
    :param num_bandits: number of bandits
    :param num_steps: how many steps to run in total
    :param figure_flag:
    :param block_flag:
    :param figure_tail_flag:
    :return:
    """
    # run the 3 strategies: random, oracle, tail.
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
        plt.show(block=block_flag)

    diff_oracle_minus_random = np.array(err_oracle) - np.array(err_random)
    rate = compute_percentage_above(err_oracle, err_random)
    return rate, diff_oracle_minus_random


def experiment_random_vs_oracle():
    # dimension of each bandit r.v.
    vec_size = 8

    # How many bandits we can sample from.
    num_bandits = 5

    # How many steps should we do for each experiment.
    num_steps = int(200)

    # repeat the experiment in order to average the results.
    num_experiments = 10

    # Show figures and whether to block in order to view the results.
    figure_flag = True
    block_flag = False

    # Create experiment name
    experiment_name_folder = f"{ROOT_DIR}/results/exp_compare_strategies_{get_time_str()}"
    print(f"Folder:{experiment_name_folder}")
    create_folder_safe(experiment_name_folder)

    # vector of results.
    rates = []
    rates_aggregate = []

    # Going over some seeds
    for seed in range(num_experiments):
        print(f"Experiment no. {seed}")
        rate, diff_oracle_minus_random = run_strategies(seed=seed,
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
