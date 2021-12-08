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

    # confidence intervals

    confidence_level = 0.95

    pre_ci = st.t.interval(alpha=confidence_level, df=len(pre) - 1, loc=np.mean(pre), scale=st.sem(pre))
    pre_ci = [round(i) for i in pre_ci]
    post_ci = st.t.interval(alpha=confidence_level, df=len(post) - 1, loc=np.mean(post), scale=st.sem(post))
    post_ci = [round(i) for i in post_ci]

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
            cs = eval(entry.split(":")[1])
            if cs:
                cs = "discard storage"
            else:
                cs = "retain storage"
        elif entry.startswith("rn:"):
            rn = eval(entry.split(":")[1])
            if rn:
                rn = "discard networks"
            else:
                rn = "retain networks"

    print("algo:", algo)
    print("env:", env)
    print("setting: {}, {}".format(rn, cs))

    # confidence intervals
    print("pre-fault CI:", pre_ci)
    print("post-fault CI:", post_ci)

    # accept/reject null
    if not reject_null:
        print("accept null ---> (pre <= post)\n")
    else:
        print("reject null ---> (pre > post)\n")


def compute_earliest_adaptation_stats(dir_):

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
            cs = eval(entry.split(":")[1])
            if cs:
                cs = "discard storage"
            else:
                cs = "retain storage"
        elif entry.startswith("rn:"):
            rn = eval(entry.split(":")[1])
            if rn:
                rn = "discard networks"
            else:
                rn = "retain networks"

    pre = []

    for seed in range(0, 30):
        dir1 = os.path.join(dir_, "seed" + str(seed))
        if os.path.exists(dir1):
            eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            pre.append(eval_data[prefault_min:prefault_max]["average_return"].values.tolist())
        else:
            print(colored("missing" + dir1, "red"))

    pre = np.array(pre).flatten()

    for post_index in range(201, 401-9+1):

        postfault_min = post_index
        postfault_max = post_index + 10

        post = []

        for seed in range(0, 30):
            dir1 = os.path.join(dir_, "seed" + str(seed))
            if os.path.exists(dir1):
                eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
                eval_data = pd.read_csv(eval_data_dir)
                post.append(eval_data[postfault_min:postfault_max]["average_return"].values.tolist())

        post = np.array(post).flatten()

        # one-tailed Welch’s t-test (do not assume equal variances)
        # H0: pre <= post
        # H1: pre > post

        t, p = ttest_ind(pre, post, equal_var=False, alternative="greater")

        alpha = 0.05

        reject_null = None

        if p / 2 < alpha:
            reject_null = True
        else:
            reject_null = False

        if not reject_null:

            print("algo:", algo)
            print("env:", env)
            print("setting: {}, {}".format(rn, cs))

            # confidence intervals

            confidence_level = 0.95

            pre_ci = st.t.interval(alpha=confidence_level, df=len(pre) - 1, loc=np.mean(pre), scale=st.sem(pre))
            pre_ci = [round(i) for i in pre_ci]
            post_ci = st.t.interval(alpha=confidence_level, df=len(post) - 1, loc=np.mean(post), scale=st.sem(post))
            post_ci = [round(i) for i in post_ci]

            print("pre-fault CI:", pre_ci)
            print("post-fault CI:", post_ci)

            # accept null
            print("accept null ---> (pre <= post)")

            # interval (time steps))

            interval_length = 0

            if algo == "SACv2":
                if env.startswith("AntEnv"):
                    interval_length = 100000
                elif env.startswith("FetchReachEnv"):
                    interval_length = 10000
            elif algo == "PPOv2":
                if env.startswith("AntEnv"):
                    interval_length = 3000000
                elif env.startswith("FetchReachEnv"):
                    interval_length = 30000
            else:
                pass

            interval_min = (postfault_min - 1) * interval_length
            interval_max = (postfault_max - 1) * interval_length
            print("interval (time steps): " + str([interval_min, interval_max]) + "\n")

            break

        # reached the end of the data array
        if postfault_max == 402:

            print("algo:", algo)
            print("env:", env)
            print("cs:", cs)
            print("rn:", rn)

            # reject null
            print("reject null ---> (pre > post)\n")


if __name__ == "__main__":

    dir = os.getcwd()

    data_dir = "/Users/sheilaannschoepp/Dropbox/Mac/Documents/openai/data"

    with open("data/stats.txt", "w") as f:
        sys.stdout = f

        prefault_min = 191
        prefault_max = 201

        postfault_min = 392
        postfault_max = 402

        print("----------------------------------------------------------\n")
        print("complete adaptation (convergence) stats\n")
        print("note: This compares the 10 evaluations prior to fault onset\nto the final 10 evaluations after fault onset\n")
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

        print("----------------------------------------------------------\n")
        print("earliest adaptation stats\n")
        print("note: This compares the 10 evaluations prior to fault onset\n"
              "to each set of 10 evaluations after fault onset.  This finds\n"
              "the first set of evaluations for which the null hypothesis\n"
              "is accepted\n")
        print("----------------------------------------------------------\n")

        ant_data_dir = os.path.join(data_dir, "ant", "exps")

        for dir1 in os.listdir(ant_data_dir):
            dir1 = os.path.join(ant_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_earliest_adaptation_stats(dir3)

        fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

        for dir1 in os.listdir(fetchreach_data_dir):
            dir1 = os.path.join(fetchreach_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_earliest_adaptation_stats(dir3)





