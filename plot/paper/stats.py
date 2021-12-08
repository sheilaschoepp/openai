import numpy as np
import os
import pandas as pd
import scipy.stats as st
import sys

from scipy.stats import ttest_ind
from termcolor import colored


def compute_complete_adaptation_stats(dir_):

    pre = []
    post = []

    for seed in range(0, 30):
        dir1 = os.path.join(dir_, "seed" + str(seed))
        if os.path.exists(dir1):
            eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            pre.append(eval_data[prefault_min:prefault_max]["average_return"].values.tolist())
            post.append(eval_data[postfault_min:postfault_max]["average_return"].values.tolist())
        else:
            print(colored("missing" + dir1, "red"))

    pre = np.array(pre).flatten()
    post = np.array(post).flatten()

    # confidence interval

    confidence_level = 0.95

    pre_ci = st.t.interval(alpha=confidence_level, df=len(pre) - 1, loc=np.mean(pre), scale=st.sem(pre))
    post_ci = st.t.interval(alpha=confidence_level, df=len(post) - 1, loc=np.mean(post), scale=st.sem(post))

    # one-tailed Welch’s t-test (do not assume equal variances)
    # H0: pre <= post
    # H1: pre > post

    t, p = ttest_ind(pre, post, equal_var=False, alternative="greater")

    alpha = 0.05

    reject_null = None

    if p/2 < alpha:
        reject_null = True
    else:
        reject_null = False

    # experiment info
    info = dir_.split("/")[-1].split("_")

    algo = None
    env = None
    cs = None  # clear storage
    rn = None  # reinitialize networks

    for entry in info:
        if entry.startswith("SAC") or entry.startswith("PPO"):
            algo = entry
        elif entry.startswith("AntEnv") or entry.startswith("FetchReach"):
            env = entry.split(":")[0]
        elif entry.startswith("crb:") or entry.startswith("cm:"):
            cs = entry.split(":")[1]
        elif entry.startswith("rn:"):
            rn = entry.split(":")[1]

    print("algo:", algo)
    print("env:", env)
    print("cs:", cs)
    print("rn:", rn)

    # confidence intervals
    print("pre_ci:", pre_ci)
    print("post_ci:", post_ci)

    # accept/reject null
    if not reject_null:
        print("accept null ---> (pre <= post)\n")
    else:
        print("reject null ---> (pre > post)\n")


def compute_adaptation_time(dir_):

    pre = []

    for seed in range(0, 30):
        dir1 = os.path.join(dir_, "seed" + str(seed))
        if os.path.exists(dir1):
            eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            pre.append(eval_data[191:201]["average_return"].values.tolist())
        else:
            print(colored("missing" + dir1, "red"))

    pre = np.array(pre).flatten()

    for post_index in range(201, 401-9+1):
        post = []
        postfault_min = post_index
        postfault_max = post_index + 10

        for seed in range(0, 30):
            dir1 = os.path.join(dir_, "seed" + str(seed))
            if os.path.exists(dir1):
                eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
                eval_data = pd.read_csv(eval_data_dir)
                post.append(eval_data[postfault_min:postfault_max]["average_return"].values.tolist())

        post = np.array(post).flatten()

        t, p = ttest_ind(pre, post, equal_var=False)

        alpha = 0.05

        reject_null = None

        # null hypothesis: the two means are equal
        # alternative hypothesis: the two means are unequal

        if p < alpha:
            reject_null = True
        else:
            reject_null = False

        if not reject_null:
            print(dir_)
            print("accept null index:", postfault_min)
            break


if __name__ == "__main__":

    data_dir = "/Users/sheilaannschoepp/Dropbox/Mac/Documents/openai/data"

    with open("data/stats.txt", "w") as f:
        sys.stdout = f

        prefault_min = 191
        prefault_max = 201

        postfault_min = 392
        postfault_max = 402

        print("----------------------------------------------------------\n")
        print("complete adaptation stats\n")
        print("----------------------------------------------------------\n")

        ant_data_dir = os.path.join(data_dir, "ant", "exps")

        for dir1 in os.listdir(ant_data_dir):
            dir1 = os.path.join(ant_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_complete_adaptation_stats(dir3)

        fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

        for dir1 in os.listdir(fetchreach_data_dir):
            dir1 = os.path.join(fetchreach_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_complete_adaptation_stats(dir3)





