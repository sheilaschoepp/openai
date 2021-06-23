import numpy as np
import pandas as pd


def param_search(param_search_seed):
    """
    Conduct a random parameter search for PPOv2 using parameter ranges from https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe.
    """

    np.random.seed(param_search_seed)

    num_samples = np.random.randint(32, 5001)  # upper bound excluded

    mini_batch_size = int(np.random.choice([2**x for x in range(2, 13)]))  # powers of two
    while mini_batch_size > num_samples:
        mini_batch_size = int(np.random.choice([2 ** x for x in range(2, 13)]))

    epochs = np.random.randint(3, 31)  # upper bound excluded

    epsilon = float(np.random.choice([0.1, 0.2, 0.3]))

    gamma = round(np.random.uniform(0.8, 0.9997), 4)

    gae_lambda = round(np.random.uniform(0.9, 1.0), 4)

    vf_loss_coef = float(np.random.choice([0.5, 1.0]))

    policy_entropy_coef = round(np.random.uniform(0, 0.01), 4)

    lr = round(np.random.uniform(0.000005, 0.006), 6)

    param = [param_search_seed,
             num_samples,
             mini_batch_size,
             epochs,
             epsilon,
             gamma,
             gae_lambda,
             vf_loss_coef,
             policy_entropy_coef,
             lr]

    return param


if __name__ == "__main__":

    params = []

    # params.append(["pss", "num_samples", "mini_batch_size", "epochs", "epsilon", "gamma", "gae_lambda", "vf_loss_coef", "policy_entropy_coef", "lr"])

    for seed in np.arange(0, 51):

        params.append(param_search(seed))

    df = pd.DataFrame(params, columns=["pss", "num_samples", "mini_batch_size", "epochs", "epsilon", "gamma", "gae_lambda", "vf_loss_coef", "policy_entropy_coef", "lr"])

    df.to_csv("ppo_hyperparameters.csv")