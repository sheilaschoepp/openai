import numpy as np
import os
import pandas as pd
import scipy.stats as st
import sys

from scipy.stats import ttest_ind
from termcolor import colored


def confidence_intervals(dir_):

    pre = []
    post = []

    for seed in range(0, 30):
        dir1 = os.path.join(dir_, "seed" + str(seed))
        if os.path.exists(dir1):
            eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            pre.append(eval_data[191:201]["average_return"].values.tolist())
            post.append(eval_data[postfault_min:postfault_max]["average_return"].values.tolist())
        else:
            print(colored("missing" + dir1, "red"))

    pre = np.array(pre).flatten()
    post = np.array(post).flatten()

    # CI

    pre_ci = st.t.interval(alpha=0.95, df=len(pre) - 1, loc=np.mean(pre), scale=st.sem(pre))
    post_ci = st.t.interval(alpha=0.95, df=len(post) - 1, loc=np.mean(post), scale=st.sem(post))

    # t test

    t, p = ttest_ind(pre, post, equal_var=False)

    alpha = 0.05

    reject_null = None

    # null hypothesis: the two means are equal
    # alternative hypothesis: the two means are unequal

    if p < alpha:
        reject_null = True
    else:
        reject_null = False

    print(dir_.split("/")[-1])
    print("pre_ci:", pre_ci)
    print("post_ci:", post_ci)
    print("accept null:", not reject_null)


def student_t_test(dir_):

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

    data_dir = "/Users/sheilaschoepp/Dropbox/Mac/Documents/openai/data"

    with open("data/t_test.txt", "w") as f:
        sys.stdout = f

        prefault_min = 191
        prefault_max = 201

        postfault_min = 392
        postfault_max = 402

        ant_data_dir = os.path.join(data_dir, "ant", "exps")

        for dir1 in os.listdir(ant_data_dir):
            dir1 = os.path.join(ant_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    confidence_intervals(dir3)

        fetchreach_data_dir = os.path.join(data_dir, "fetchreach", "exps")

        for dir1 in os.listdir(fetchreach_data_dir):
            dir1 = os.path.join(fetchreach_data_dir, dir1)
            for dir2 in os.listdir(dir1):
                dir2 = os.path.join(dir1, dir2)
                for dir3 in os.listdir(dir2):
                    dir3 = os.path.join(dir2, dir3)
                    confidence_intervals(dir3)





