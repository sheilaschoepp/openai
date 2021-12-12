import numpy as np
import os
import pandas as pd
import scipy.stats as st
import sys

from scipy.stats import sem, ttest_ind
from termcolor import colored


def compute_complete_adaptation_stats(dir_):

    # experiment info
    info = dir_.split("/")[-1].split("_")

    algo = None
    env = None
    rn = None  # reinitialize networks
    cs = None  # clear storage

    for entry in info:
        if entry.startswith("SAC") or entry.startswith("PPO"):
            algo = entry
        elif (entry.startswith("AntEnv") or entry.startswith("FetchReach")) and not entry.startswith("FetchReachEnv-v0"):
            env = entry.split(":")[0]
        elif entry.startswith("rn:"):
            rn = eval(entry.split(":")[1])
        elif entry.startswith("crb:") or entry.startswith("cm:"):
            cs = eval(entry.split(":")[1])

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

    complete_adaptation_data.append([algo, env, rn, cs, post])

    # confidence intervals

    pre_ci = st.t.interval(alpha=confidence_level, df=len(pre) - 1, loc=np.mean(pre), scale=st.sem(pre))
    if env.startswith("AntEnv"):
        pre_ci = [round(i) for i in pre_ci]
    else:
        pre_ci = [round(i, 3) for i in pre_ci]
    post_ci = st.t.interval(alpha=confidence_level, df=len(post) - 1, loc=np.mean(post), scale=st.sem(post))
    if env.startswith("AntEnv"):
        post_ci = [round(i) for i in post_ci]
    else:
        post_ci = [round(i, 3) for i in post_ci]

    # one-tailed Welch’s t-test (do not assume equal variances)
    # H0: pre_mean <= post_mean
    # H1: pre_mean > post_mean

    t, p = ttest_ind(pre, post, equal_var=False, alternative="greater")
    p = p / 2  # one-tailed test

    reject_null = None

    if p < alpha:
        reject_null = True
    else:
        reject_null = False

    print("algo:", algo)
    print("env:", env)

    if rn:
        rn = "discard networks"
    else:
        rn = "retain networks"

    if cs:
        cs = "discard storage"
    else:
        cs = "retain storage"

    print("setting: {}, {}".format(rn, cs))

    pre_mean = np.mean(pre)
    if env.startswith("AntEnv"):
        pre_mean = round(pre_mean)
    else:
        pre_mean = round(pre_mean, 3)

    pre_sem = sem(pre)
    if env.startswith("AntEnv"):
        pre_sem = round(pre_sem)
    else:
        pre_sem = round(pre_sem, 3)

    post_mean = np.mean(post)
    if env.startswith("AntEnv"):
        post_mean = round(post_mean)
    else:
        post_mean = round(post_mean, 3)

    post_sem = sem(post)
    if env.startswith("AntEnv"):
        post_sem = round(post_sem)
    else:
        post_sem = round(post_sem, 3)

    # stats
    print("pre-fault mean:", pre_mean)
    print("pre-fault sem:", pre_sem)
    print("pre-fault CI:", pre_ci)

    print("post-fault mean:", post_mean)
    print("post-fault sem:", post_sem)
    print("post-fault CI:", post_ci)

    print("t:", round(t, 3))
    print("p:", round(p, 3))

    # accept/reject null
    if not reject_null:
        print("accept null ---> (pre_mean <= post_mean)\n")
    else:
        print("reject null ---> (pre_mean > post_mean)\n")


def compute_complete_adaptation_setting_comparison_stats():

    # create a list of algorithms and environments
    algorithms = []
    envs = []

    for entry in complete_adaptation_data:
        algo = entry[0]
        env = entry[1]
        if algo not in algorithms:
            algorithms.append(algo)
        if env not in envs:
            envs.append(env)

    # compare settings
    for algo in algorithms:
        for env in envs:

            # obtain data for all settings for a specific algorithm and environment

            algo_ = None
            env_ = None

            setting_data = []

            for entry in complete_adaptation_data:
                algo_ = entry[0]
                env_ = entry[1]
                if algo == algo_ and env == env_:
                    rn = entry[2]
                    cs = entry[3]
                    data = entry[4]
                    setting_data.append([rn, cs, data])

            assert(len(setting_data) == 4)  # check

            setting_data.sort()

            print("------------------------\n")
            print("algo:", algo)
            print("env:", env, "\n")
            print("------------------------\n")

            def compute_t_test(a, b):

                a_rn = a[0]
                a_cs = a[1]
                a_data = a[2]

                b_rn = b[0]
                b_cs = b[1]
                b_data = b[2]

                # two-tailed Welch’s t-test (do not assume equal variances)
                # H0: a_mean = b_mean
                # H1: a_mean != b_mean

                t, p = ttest_ind(a_data, b_data, equal_var=False)

                reject_null = None

                if p < alpha:
                    reject_null = True
                else:
                    reject_null = False

                if a_rn:
                    a_rn = "discard networks"
                else:
                    a_rn = "retain networks"

                if b_rn:
                    b_rn = "discard networks"
                else:
                    b_rn = "retain networks"

                if a_cs:
                    a_cs = "discard storage"
                else:
                    a_cs = "retain storage"

                if b_cs:
                    b_cs = "discard storage"
                else:
                    b_cs = "retain storage"

                print("a setting: {}, {}".format(a_rn, a_cs))
                print("b setting: {}, {}".format(b_rn, b_cs))

                print("t:", round(t, 3))
                print("p:", round(p, 3))

                # accept/reject null
                if not reject_null:
                    print("accept null ---> (a_mean = b_mean)\n")
                else:
                    print("reject null ---> (a_mean != b_mean)\n")

            for i in range(1, 4):

                a_ = setting_data[0]
                b_ = setting_data[i]

                compute_t_test(a_, b_)

            for i in range(2, 4):

                a_ = setting_data[1]
                b_ = setting_data[i]

                compute_t_test(a_, b_)

            for i in range(3, 4):

                a_ = setting_data[2]
                b_ = setting_data[i]

                compute_t_test(a_, b_)


def compute_earliest_adaptation_stats(dir_):

    # experiment info
    info = dir_.split("/")[-1].split("_")

    algo = None  # algorithm
    env = None  # environment
    rn = None  # reinitialize networks
    cs = None  # clear storage
    tef = None  # time step evaluation frequency
    nt = None  # normal time steps

    for entry in info:
        if entry.startswith("SAC") or entry.startswith("PPO"):
            algo = entry
        elif (entry.startswith("AntEnv") or entry.startswith("FetchReach")) and not entry.startswith("FetchReachEnv-v0"):
            env = entry.split(":")[0]
        elif entry.startswith("rn:"):
            rn = eval(entry.split(":")[1])
        elif entry.startswith("crb:") or entry.startswith("cm:"):
            cs = eval(entry.split(":")[1])
        elif entry.startswith("tef:"):
            tef = int(entry.split(":")[1])
        elif entry.startswith("Ant-v2") or entry.startswith("FetchReachEnv-v0"):
            nt = int(entry.split(":")[1])

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
        # H0: pre_mean <= post_mean
        # H1: pre_mean > post_mean

        t, p = ttest_ind(pre, post, equal_var=False, alternative="greater")
        p = p / 2  # one-tailed test

        reject_null = None

        if p < alpha:
            reject_null = True
        else:
            reject_null = False

        if not reject_null:

            print("algo:", algo)
            print("env:", env)

            # confidence intervals

            pre_ci = st.t.interval(alpha=confidence_level, df=len(pre) - 1, loc=np.mean(pre), scale=st.sem(pre))
            if env.startswith("AntEnv"):
                pre_ci = [round(i) for i in pre_ci]
            else:
                pre_ci = [round(i, 3) for i in pre_ci]
            post_ci = st.t.interval(alpha=confidence_level, df=len(post) - 1, loc=np.mean(post), scale=st.sem(post))
            if env.startswith("AntEnv"):
                post_ci = [round(i) for i in post_ci]
            else:
                post_ci = [round(i, 3) for i in post_ci]

            # mean

            pre_mean = np.mean(pre)
            if env.startswith("AntEnv"):
                pre_mean = round(pre_mean)
            else:
                pre_mean = round(pre_mean, 3)

            pre_sem = sem(pre)
            if env.startswith("AntEnv"):
                pre_sem = round(pre_sem)
            else:
                pre_sem = round(pre_sem, 3)

            post_mean = np.mean(post)
            if env.startswith("AntEnv"):
                post_mean = round(post_mean)
            else:
                post_mean = round(post_mean, 3)

            post_sem = sem(post)
            if env.startswith("AntEnv"):
                post_sem = round(post_sem)
            else:
                post_sem = round(post_sem, 3)

            # interval (time steps)

            interval_min = (postfault_min - 1) * tef
            interval_max = (postfault_max - 1) * tef

            # real time (hours)

            real_time_per_eval = None
            if env.startswith("AntEnv"):
                if env == "AntEnv-v1":
                    if algo.startswith("SAC"):
                        if rn and cs:
                            init = 24
                            first = (14 * 60 + 26) - init
                            second = (28 * 60 + 41) - init - first
                            third = (42 * 60 + 58) - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds)
                    else:
                        if rn and cs:
                            init = 2
                            first = (30 * 60 + 4) - init
                            second = (59 * 60 + 53) - init - first
                            third = (89 * 60 + 31) - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds)
                        elif rn and not cs:
                            init = 2
                            first = (29 * 60 + 39) - init
                            second = (58 * 60 + 59) - init - first
                            third = (87 * 60 + 55) - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds)
                elif env == "AntEnv-v2":
                    if algo.startswith("SAC"):
                        if rn and cs:
                            init = 21
                            first = (14 * 60 + 16) - init
                            second = (28 * 60 + 26) - init - first
                            third = (42 * 60 + 34) - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds)
                    else:
                        real_time_per_eval = 0
                elif env == "AntEnv-v3":
                    if algo.startswith("SAC"):
                        real_time_per_eval = 0
                    else:
                        real_time_per_eval = 0
                elif env == "AntEnv-v4":
                    if algo.startswith("SAC"):
                        real_time_per_eval = 0
                    else:
                        if rn and cs:
                            init = 0
                            first = 0 - init
                            second = 0 - init - first
                            third = 0 - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds) TODO
                        elif rn and not cs:
                            init = 0
                            first = 0 - init
                            second = 0 - init - first
                            third = 0 - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds) TODO
            else:
                if env == "FetchReachEnv-v4":
                    if algo.startswith("SAC"):
                        real_time_per_eval = 0
                    else:
                        real_time_per_eval = 0
                elif env == "FetchReachEnv-v6":
                    if algo.startswith("SAC"):
                        real_time_per_eval = 0
                    else:
                        real_time_per_eval = 0

            real_time = real_time_per_eval * (interval_max - nt) / tef
            real_time = real_time / (60 * 60)  # hours
            real_time = round(real_time, 2)

            if rn:
                rn = "discard networks"
            else:
                rn = "retain networks"

            if cs:
                cs = "discard storage"
            else:
                cs = "retain storage"

            print("setting: {}, {}".format(rn, cs))

            # stats
            print("pre-fault mean:", pre_mean)
            print("pre-fault sem:", pre_sem)
            print("pre-fault CI:", pre_ci)

            print("post-fault mean:", post_mean)
            print("post-fault sem:", post_sem)
            print("post-fault CI:", post_ci)

            print("t:", round(t, 3))
            print("p:", round(p, 3))

            # accept null
            print("accept null ---> (pre_mean <= post_mean)")

            # interval
            print("interval (time steps): " + str([interval_min, interval_max]))

            # real time
            print("real time to reach interval max (hours):", real_time, "\n")

            break

        # reached the end of the data array
        if postfault_max == 402:

            # print("algo:", algo)
            # print("env:", env)
            #
            # if rn:
            #     rn = "discard networks"
            # else:
            #     rn = "retain networks"
            #
            # if cs:
            #     cs = "discard storage"
            # else:
            #     cs = "retain storage"
            #
            # print("rn:", rn)
            # print("cs:", cs)
            #
            # # reject null
            # print("reject null ---> (pre_mean > post_mean)\n")

            pass


if __name__ == "__main__":

    data_dir = os.getcwd().split("/")[:-2]
    data_dir.append("data")
    data_dir = "/".join(data_dir)

    prefault_min = 191
    prefault_max = 201

    postfault_min = 392
    postfault_max = 402

    confidence_level = 0.95
    alpha = 0.05

    complete_adaptation = False
    earliest_adaptation = True

    if complete_adaptation:

        # list of list: entries [algo, env, rn, cs, post]
        complete_adaptation_data = []

        with open("stats/complete_adaptation_stats.txt", "w") as f:

            sys.stdout = f

            print("----------------------------------------------------------\n")
            print("complete adaptation (convergence) stats\n")
            print("note: This compares the 10 evaluations prior to fault onset\n"
                  "to the final 10 evaluations after fault onset\n")
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

        with open("stats/complete_adaptation_setting_comparison_stats.txt", "w") as f:

            sys.stdout = f

            print("----------------------------------------------------------\n")
            print("complete adaptation (convergence) setting comparison stats\n")
            print("note: This compares the final 10 evaluations after fault\n"
                  "onset across all setting pairs.\n")
            print("----------------------------------------------------------\n")

            compute_complete_adaptation_setting_comparison_stats()

    if earliest_adaptation:

        with open("stats/earliest_adaptation_stats.txt", "w") as f:

            sys.stdout = f

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





