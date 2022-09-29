from src.bandits.bandits_estimator import BanditsEstimator
from src.bandits.bandits import Bandits


def run_random_strategy(seed,
                        vec_size,
                        num_bandits,
                        num_steps):
    """
    Random strategy
    ---------------
    Estimating the bandits distribution where in each turn we choose randomly a single bandit and get a sample from it.
    :param seed: seed for reproducability...
    :param vec_size: dimension of the distribution
    :param num_bandits: num of bandits
    :param num_steps: num of steps
    :return:
    """
    # initiate bandits
    bandits = Bandits(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    estimator = BanditsEstimator(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    err = []  # the error according to the steps.
    for i in range(num_steps):
        # Get the next bandit according to uniform distribution (this is the random strategy).
        bandit_index = estimator.get_next_bandit_random()

        # Get the sample
        sample = bandits.get_sample(bandit_index=bandit_index)

        # Estimates the stats
        estimator.update_stats(bandit_index=bandit_index,  # The bandit index
                               sample=sample,  # the sample
                               true_stats=bandits.bandits_prob,  # just for the estimation, what is true probability
                               )
        err.append(estimator.total_error)
        # print(estimator.to_string())
    return err


def run_oracle_strategy(seed,
                        vec_size,
                        num_bandits,
                        num_steps,
                        step_print=10):
    """
    Oracle Strategy is a strategy where we assume that we have the true distribution (which in general is not true
    of course). By that, we assume having the true probability, and we choose the bandit where we have the worst
    estimation so far.
    :params - SIMILARLY to random strategy.
    :param step_print: just from printing outside
    :return:
    """
    bandits = Bandits(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    estimator = BanditsEstimator(seed=seed, vec_size=vec_size, num_bandits=num_bandits)
    err = []
    for i in range(num_steps):
        if i % step_print == 0:
            print(f"step run_oracle_strategy {i}")
        # The following is how we choose
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
