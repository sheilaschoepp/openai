import os
import pandas as pd
import numpy as np


def check_recovery(directory):

    for dir_ in os.listdir(directory):

        # dir_ has format: SACv2_FetchReachEnvGE-v1:2000000_FetchReachEnvGE-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda
        parameters = dir_.split("_")

        # obtain setting
        mem = None
        rn = None
        for p in parameters:
            if "crb:" in p or "cm:" in p:
                mem = eval(p.split(":")[1])
            elif "rn:" in p:
                rn = eval(p.split(":")[1])

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
        df_var = df.var()

        # check difference between pre-fault and post-recovery performance

        prefault = df[0:201]
        prefault = prefault.iloc[-1:, 1:]

        postfault = df[201:]

        print(1)


if __name__ == "__main__":

    # local for Ant PPO
    # ppo_data_dir = "/media/sschoepp/easystore/shared/ant/faulty/ppo"
    ppo_data_dir = "/mnt/DATA/shared/ant/faulty/ppo"

    # v1

    directory = os.path.join(ppo_data_dir, "v1")
    check_recovery(directory)

    print(1)
