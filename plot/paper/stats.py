import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as st
import seaborn as sns
import sys

from PIL import Image
from scipy.stats import sem, ttest_ind
from termcolor import colored

sns.set_theme()
palette_colours = ["#0173b2", "#A63F93", "#027957", "#AD4B00", "#000000"]

LARGE = 16
MEDIUM = 14

plt.rc("axes", titlesize=LARGE)     # fontsize of the axes title
plt.rc("axes", labelsize=LARGE)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM)   # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM)   # fontsize of the tick labels
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"


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


def compute_complete_adaptation_comparison_stats():

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
                            init = 1
                            first = (30 * 60 + 6) - init
                            second = (60 * 60 + 16) - init - first
                            third = (90 * 60 + 9) - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds)
                        elif rn and not cs:
                            init = 1
                            first = (30 * 60 + 16) - init
                            second = (60 * 60 + 27) - init - first
                            third = (90 * 60 + 27) - init - first - second
                            real_time_per_eval = (first + second + third) / 3  # average (seconds)
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


def compute_performance_drop_stats(dir_):

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
            pre.append(eval_data[prefault_min + 0:prefault_max]["average_return"].values.tolist())
            post.append(eval_data[prefault_max:prefault_max + 10]["average_return"].values.tolist())
        else:
            print(colored("missing" + dir1, "red"))

    pre = np.array(pre).flatten()
    post = np.array(post).flatten()

    drops = []
    for i in range(len(pre)):
        diff = post[i] - pre[i]
        drops.append(diff)

    performance_drop_data.append([algo, env, rn, cs, drops])

    drop_ci = st.t.interval(alpha=confidence_level, df=len(pre) - 1, loc=np.mean(drops), scale=st.sem(drops))
    if env.startswith("AntEnv"):
        drop_ci = [round(i) for i in drop_ci]
    else:
        drop_ci = [round(i, 3) for i in drop_ci]

    drop_mean = np.mean(drops)
    if env.startswith("AntEnv"):
        drop_mean = round(drop_mean)
    else:
        drop_mean = round(drop_mean, 3)

    drop_sem = sem(drops)
    if env.startswith("AntEnv"):
        drop_sem = round(drop_sem)
    else:
        drop_sem = round(drop_sem, 3)

    if rn:
        rn = "discard networks"
    else:
        rn = "retain networks"

    if cs:
        cs = "discard storage"
    else:
        cs = "retain storage"

    print("algo:", algo)
    print("env:", env)
    print("setting: {}, {}".format(rn, cs))

    # stats
    print("drop mean:", drop_mean)
    print("drop sem:", drop_sem)
    print("drop CI:", drop_ci, "\n")


def compute_performance_drop_comparison_stats():

    # create a list of algorithms and environments
    algorithms = []
    envs = []

    for entry in performance_drop_data:
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

            for entry in performance_drop_data:
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


def compute_postfault_performance_drop(dir_):

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
    asymp = []

    for seed in range(0, 30):
        dir1 = os.path.join(dir_, "seed" + str(seed))
        if os.path.exists(dir1):
            eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            pre.append(eval_data[prefault_min:prefault_max]["average_return"].values.tolist())
            post.append(eval_data[postfault_min:postfault_max]["average_return"].values.tolist())
            asymp.append(eval_data[-10:]["average_return"].values.tolist())
        else:
            print(colored("missing" + dir1, "red"))

    pre = np.array(pre).flatten()
    post = np.array(post).flatten()
    asymp = np.array(asymp).flatten()

    pre_mean = np.mean(pre)
    post_mean = np.mean(post)
    asymp_mean = np.mean(asymp)

    post_sem = sem(post)

    CI_Z = 1.960

    if env.startswith("Ant"):
        normal_env = "Ant-v2"
        pre_mean = round(pre_mean)
        post_mean = round(post_mean)
        post_ci = round(post_sem * CI_Z)
    elif env.startswith("FetchReach"):
        normal_env = "FetchReach-v1"
        pre_mean = round(pre_mean, 3)
        post_mean = round(post_mean, 3)
        post_ci = round(post_sem * CI_Z, 3)

    prefault_performance_data[algo + ", " + normal_env] = pre_mean
    postfault_performance_data.append([algo, env, rn, cs, post_mean, post_ci, asymp_mean])


def plot_postfault_performance_drop(interval):

    algos = ["PPOv2", "SACv2"]
    envs = ["Ant", "FetchReach"]

    # dictionary: entries {"algo, env": mean}
    # prefault_performance_data = {}
    # list of list: entries [algo, env, rn, cs, mean]
    # postfault_performance_data = []

    for algo in algos:
        for env in envs:

            data = []

            if env == "Ant":
                pre = prefault_performance_data[algo + ", " + "Ant-v2"]
            elif env == "FetchReach":
                pre = prefault_performance_data[algo + ", " + "FetchReach-v1"]

            rnFcsF = None
            rnFcsT = None
            rnTcsF = None
            rnTcsT = None
            rnFcsF_ci = None
            rnFcsT_ci = None
            rnTcsF_ci = None
            rnTcsT_ci = None

            CI_Z = 1.960

            asymp_mean_baseline = None
            for entry in postfault_performance_data:
                algo_ = entry[0]
                env_ = entry[1]
                if algo_.startswith(algo) and env_.startswith(env):
                    rn = entry[2]
                    cs = entry[3]
                    post_mean = entry[4]
                    post_ci = entry[5]
                    asymp_mean = entry[6]
                    if not rn and not cs:
                        rnFcsF = post_mean
                        rnFcsF_ci = post_ci
                    elif not rn and cs:
                        rnFcsT = post_mean
                        rnFcsT_ci = post_ci
                    elif rn and not cs:
                        rnTcsF = post_mean
                        rnTcsF_ci = post_ci
                    elif rn and cs:
                        rnTcsT = post_mean
                        rnTcsT_ci = post_ci
                        asymp_mean_baseline = asymp_mean
                    if rnFcsF and rnFcsT and rnTcsF and rnTcsT:
                        data.append([algo_, env_, rnFcsF, rnFcsT, rnTcsF, rnTcsT, rnFcsF_ci, rnFcsT_ci, rnTcsF_ci, rnTcsT_ci, asymp_mean_baseline])

                        rnFcsF = None
                        rnFcsT = None
                        rnTcsF = None
                        rnTcsT = None
                        rnFcsF_ci = None
                        rnFcsT_ci = None
                        rnTcsF_ci = None
                        rnTcsT_ci = None

            # plot
            if env == "Ant":

                # reorganize data to ["AntEnv-v2", "AntEnv-v3", "AntEnv-v1", "AntEnv-v4"]
                data_ = [data[1], data[2], data[0], data[3]]
                labels = ["hip ROM\nrestriction", "ankle ROM\nrestriction", "broken,\nsevered limb", "broken,\nunsevered limb"]

                # reorganize the data
                # labels = []
                rnFcsFs = []
                rnFcsTs = []
                rnTcsFs = []
                rnTcsTs = []
                rnFcsFs_sem = []
                rnFcsTs_sem = []
                rnTcsFs_sem = []
                rnTcsTs_sem = []
                asymp_mean_baselines = []
                for entry in data_:
                    # labels.append(entry[1])
                    rnFcsFs.append(entry[2])
                    rnFcsTs.append(entry[3])
                    rnTcsFs.append(entry[4])
                    rnTcsTs.append(entry[5])
                    rnFcsFs_sem.append(entry[6])
                    rnFcsTs_sem.append(entry[7])
                    rnTcsFs_sem.append(entry[8])
                    rnTcsTs_sem.append(entry[9])
                    asymp_mean_baselines.append(entry[10])

                bar_width = 0.2
                br1 = np.arange(len(labels))
                br2 = [x + bar_width for x in br1]
                br3 = [x + bar_width for x in br2]
                br4 = [x + bar_width for x in br3]

                # plt.axhline(y=0, color=palette_colours[4], linewidth=1)

                plt.bar(br1, np.array(rnFcsFs), yerr=rnFcsFs_sem, color=palette_colours[1], width=bar_width)
                plt.bar(br2, np.array(rnFcsTs), yerr=rnFcsTs_sem, color=palette_colours[2], width=bar_width)
                plt.bar(br3, np.array(rnTcsFs), yerr=rnTcsFs_sem, color=palette_colours[3], width=bar_width)
                plt.bar(br4, np.array(rnTcsTs), yerr=rnTcsTs_sem, color=palette_colours[4], width=bar_width)

                plt.axhline(xmin=0.05, xmax=0.23, y=asymp_mean_baselines[0], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.29, xmax=0.47, y=asymp_mean_baselines[1], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.53, xmax=0.71, y=asymp_mean_baselines[2], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.77, xmax=0.95, y=asymp_mean_baselines[3], color=palette_colours[4], linestyle="dashed", linewidth=1)

                plt.axvline(x=0.3, color="black", ymax=0.025)
                plt.axvline(x=1.3, color="black", ymax=0.025)
                plt.axvline(x=2.3, color="black", ymax=0.025)
                plt.axvline(x=3.3, color="black", ymax=0.025)

                plt.xticks([r + 0.3 for r in range(len(labels))],  labels)
                plt.yticks([-2000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000])

            elif env == "FetchReach":

                data_ = data
                labels = ["frozen shoulder lift\nposition sensor", "elbow flex\nposition slippage"]

                # reorganize the data
                # labels = []
                rnFcsFs = []
                rnFcsTs = []
                rnTcsFs = []
                rnTcsTs = []
                rnFcsFs_sem = []
                rnFcsTs_sem = []
                rnTcsFs_sem = []
                rnTcsTs_sem = []
                asymp_mean_baselines = []
                for entry in data_:
                    # labels.append(entry[1])
                    rnFcsFs.append(entry[2])
                    rnFcsTs.append(entry[3])
                    rnTcsFs.append(entry[4])
                    rnTcsTs.append(entry[5])
                    rnFcsFs_sem.append(entry[6])
                    rnFcsTs_sem.append(entry[7])
                    rnTcsFs_sem.append(entry[8])
                    rnTcsTs_sem.append(entry[9])
                    asymp_mean_baselines.append(entry[10])

                bar_width = 0.2
                br1 = np.arange(len(labels))
                br2 = [x + bar_width for x in br1]
                br3 = [x + bar_width for x in br2]
                br4 = [x + bar_width for x in br3]

                plt.bar(br1, np.array(rnFcsFs), yerr=rnFcsFs_sem, color=palette_colours[1], width=bar_width, bottom=asymp_mean_baselines)
                plt.bar(br2, np.array(rnFcsTs), yerr=rnFcsTs_sem, color=palette_colours[2], width=bar_width, bottom=asymp_mean_baselines)
                plt.bar(br3, np.array(rnTcsFs), yerr=rnTcsFs_sem, color=palette_colours[3], width=bar_width, bottom=asymp_mean_baselines)
                plt.bar(br4, np.array(rnTcsTs), yerr=rnTcsTs_sem, color=palette_colours[4], width=bar_width, bottom=asymp_mean_baselines)

                plt.axhline(xmin=0.05, xmax=0.45, y=asymp_mean_baselines[0], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.555, xmax=0.955, y=asymp_mean_baselines[1], color=palette_colours[4], linestyle="dashed", linewidth=1)

                plt.axvline(x=0.3, color="black", ymax=0.025)
                plt.axvline(x=1.3, color="black", ymax=0.025)

                plt.xticks([r + 0.3 for r in range(len(labels))], labels)
                plt.yticks(np.arange(-30, 1, 5))
                plt.ylim((-30, 1.5))

            if algo.startswith("PPO"):
                plt.title("Proximal Policy Optimization (PPO)")
            elif algo.startswith("SAC"):
                plt.title("Soft Actor-Critic (SAC)")

            plt.xlabel("fault")
            plt.ylabel("average return (30 seeds)")
            plt.tight_layout()

            plot_directory = os.path.join(os.getcwd(), "plots", env.lower(), algo[:-2])
            os.makedirs(plot_directory, exist_ok=True)

            filename = plot_directory + "/{}_{}_average_return_after_fault_onset_{}.jpg".format(algo[:-2].upper(), env, interval)
            plt.savefig(filename, dpi=300)
            # Image.open(filename).convert("CMYK").save(filename)

            # plt.show()

            plt.close()


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
    earliest_adaptation = False
    performance_drop = False
    plot_performance_drop = True

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

        with open("stats/complete_adaptation_comparison_stats.txt", "w") as f:

            sys.stdout = f

            print("----------------------------------------------------------\n")
            print("complete adaptation (convergence) setting comparison stats\n")
            print("note: This compares the final 10 evaluations after fault\n"
                  "onset across all setting pairs.\n")
            print("----------------------------------------------------------\n")

            compute_complete_adaptation_comparison_stats()

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

    if performance_drop:

        # list of list: entries [algo, env, rn, cs, post]
        performance_drop_data = []

        with open("stats/performance_drop_stats.txt", "w") as f:

            sys.stdout = f

            print("----------------------------------------------------------\n")
            print("performance drop stats stats\n")
            print("The drop in performance is measured as the average\n"
                  "difference between the performance prior to fault onset and\n"
                  "the lowest post-fault performance over 10 evaluations.\n")
            print("----------------------------------------------------------\n")

            ant_data_dir = os.path.join(data_dir, "ant", "exps")

            for dir1 in os.listdir(ant_data_dir):
                dir1 = os.path.join(ant_data_dir, dir1)
                for dir2 in os.listdir(dir1):
                    dir2 = os.path.join(dir1, dir2)
                    for dir3 in os.listdir(dir2):
                        dir3 = os.path.join(dir2, dir3)
                        compute_performance_drop_stats(dir3)

            fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

            for dir1 in os.listdir(fetchreach_data_dir):
                dir1 = os.path.join(fetchreach_data_dir, dir1)
                for dir2 in os.listdir(dir1):
                    dir2 = os.path.join(dir1, dir2)
                    for dir3 in os.listdir(dir2):
                        dir3 = os.path.join(dir2, dir3)
                        compute_performance_drop_stats(dir3)

        with open("stats/performance_drop_comparison_stats.txt", "w") as f:

            sys.stdout = f

            print("----------------------------------------------------------\n")
            print("performance drop comparison stats\n")
            print("note: This compares the performance drop across all \n"
                  "post-fault setting pairs.\n")
            print("----------------------------------------------------------\n")

            compute_performance_drop_comparison_stats()

    if plot_performance_drop:

        # 10 evals

        prefault_min = 191
        prefault_max = 201

        postfault_min = 211
        postfault_max = 212

        eval_interval = f"[{postfault_min}:{postfault_max}]"

        # dictionary: entries {"algo, env": mean}
        prefault_performance_data = {}
        # list of list: entries [algo, env, rn, cs, mean]
        postfault_performance_data = []

        ant_data_dir = os.path.join(data_dir, "ant", "exps")

        for dir1 in os.listdir(ant_data_dir):
            dir1 = os.path.join(ant_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_postfault_performance_drop(dir3)

        fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

        for dir1 in os.listdir(fetchreach_data_dir):
            dir1 = os.path.join(fetchreach_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_postfault_performance_drop(dir3)

        postfault_performance_data.sort()
        plot_postfault_performance_drop(eval_interval)

        # no evals

        prefault_min = 191
        prefault_max = 201

        postfault_min = 201
        postfault_max = 202

        eval_interval = f"[{postfault_min}:{postfault_max}]"

        # dictionary: entries {"algo, env": mean}
        prefault_performance_data = {}
        # list of list: entries [algo, env, rn, cs, mean]
        postfault_performance_data = []

        ant_data_dir = os.path.join(data_dir, "ant", "exps")

        for dir1 in os.listdir(ant_data_dir):
            dir1 = os.path.join(ant_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_postfault_performance_drop(dir3)

        fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

        for dir1 in os.listdir(fetchreach_data_dir):
            dir1 = os.path.join(fetchreach_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_postfault_performance_drop(dir3)

        postfault_performance_data.sort()
        plot_postfault_performance_drop(eval_interval)

        # one eval

        prefault_min = 191
        prefault_max = 201

        postfault_min = 202
        postfault_max = 203

        eval_interval = f"[{postfault_min}:{postfault_max}]"

        # dictionary: entries {"algo, env": mean}
        prefault_performance_data = {}
        # list of list: entries [algo, env, rn, cs, mean]
        postfault_performance_data = []

        ant_data_dir = os.path.join(data_dir, "ant", "exps")

        for dir1 in os.listdir(ant_data_dir):
            dir1 = os.path.join(ant_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_postfault_performance_drop(dir3)

        fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

        for dir1 in os.listdir(fetchreach_data_dir):
            dir1 = os.path.join(fetchreach_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    compute_postfault_performance_drop(dir3)

        postfault_performance_data.sort()
        plot_postfault_performance_drop(eval_interval)