import os
import sys

import pandas as pd
from scipy import stats


def check_recovery(directory):

    for dir_ in os.listdir(directory):

        # if dir_[-2:] == "_r":
        #     continue

        # dir_ has format: SACv2_FetchReachEnvGE-v1:2000000_FetchReachEnvGE-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda
        parameters = dir_.split("_")

        # obtain setting
        algorithm = parameters[0]
        env_name = parameters[1].split(":")[0]
        mem = None
        rn = None
        for p in parameters:
            if "crb:" in p or "cm:" in p:
                mem = eval(p.split(":")[1])
            elif "rn:" in p:
                rn = eval(p.split(":")[1])

        print(algorithm, env_name, "mem={}".format(mem), "rn:{}".format(rn))

        # obtain data for setting: mean and standard error

        dfs = []

        data_dir = os.path.join(directory, dir_)

        for s in os.listdir(data_dir):
            seed_foldername = os.path.join(data_dir, s)
            csv_filename = os.path.join(seed_foldername, "csv", "eval_data.csv")

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]]
            dfs.append(df)

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()

        # check difference between pre-fault and post-recovery performance

        interval_size = 10

        prefault = df_mean[0:201]
        prefault_interval = prefault.iloc[-interval_size:, 1:]

        min = prefault.min()
        max = prefault_interval.mean()

        postfault = df_mean[201:]
        start = 0
        end = postfault.count()[1]

        finished = False
        while not finished:
            if start + interval_size > end:
                finished = True
                postfault_interval = postfault.iloc[-interval_size:, 1:]
            else:
                postfault_interval = postfault.iloc[start:start+interval_size, 1:]

            # welch's t-test
            # assume unequal variances
            # https://en.wikipedia.org/wiki/Welch%27s_t-test
            # https://en.wikipedia.org/wiki/Student%27s_t-distribution

            # null: postfault_interval == prefault_interval
            # alternative: postfault_interval != prefault_interval

            # significance level
            alpha = 0.05

            t, p = stats.ttest_ind(postfault_interval.values, prefault_interval.values, equal_var=False)

            # Wilcoxon Signed Rank Test
            # https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
            # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_nonparametric/BS704_Nonparametric6.html

            x = postfault_interval.reset_index().iloc[:, 1:].values.squeeze()
            y = prefault_interval.reset_index().iloc[:, 1:].values.squeeze()
            w, p = stats.wilcoxon(x, y)

            if w <= 8:
                # reject null hypothesis
                pass
            else:
                # accept null hypothesis
                pass

            if p <= alpha:
                # reject null hypothesis; evidence that performance has changed
                pass
            else:
                # accept null hypothesis; performance has not changed
                print("accept null", start, start + interval_size)
                break

            start += 1


if __name__ == "__main__":

    os.makedirs(os.path.join(os.getcwd(), "data"))

    with open("data/numerical.txt", "w") as f:
        sys.stdout = f

        # local for Ant PPO
        # ppo_data_dir = "/media/sschoepp/easystore/shared/ant/faulty/ppo"
        ppo_data_dir = "/mnt/DATA/shared/fetchreach/faulty/ppo"

        # v1

        directory = os.path.join(ppo_data_dir, "v1")
        check_recovery(directory)

        # v4

        directory = os.path.join(ppo_data_dir, "v4")
        check_recovery(directory)

        # v6

        directory = os.path.join(ppo_data_dir, "v6")
        check_recovery(directory)
