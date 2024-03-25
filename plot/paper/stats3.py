import numpy as np
import os
import pandas as pd
import sys

from scipy import stats
from termcolor import colored


if __name__ == "__main__":

    with open("stats/adaptation_speed_analysis.txt", "w") as f:

        sys.stdout = f

        print("---------------------------------------------------------\n")
        print("Adaptation Speed Analysis for Ant-v2. This analysis\n"
              "displays the time step at which the 95% confidence\n"
              "interval of the final average return of our baseline\n"
              "is reached. Methods that fail to reach this confidence\n"
              "interval are represented with a dashed line.\n")
        print("---------------------------------------------------------\n")

        # AntEnv-v1: Broken, Severed Limb

        base_path = f"/home/sschoepp/Documents/openai/data/ant/exps/complete/ppo/v1/PPOv2_AntEnv-v1:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000"

        cmF_rnF_path = f"{base_path}_cm:False_rn:False_d:cpu"
        cmF_rnT_path = f"{base_path}_cm:False_rn:True_d:cpu"
        cmT_rnF_path = f"{base_path}_cm:True_rn:False_d:cpu"
        baseline_cmT_rnT_path = f"{base_path}_cm:True_rn:True_d:cpu"

        # BASELINE cmT_rnT

        baseline_cmT_rnT_dfs = []

        for seed in range(0, 30):
            dir = os.path.join(baseline_cmT_rnT_path, "seed" + str(seed))
            if os.path.exists(dir):
                eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                eval_data_df = pd.read_csv(eval_data_dir)
                eval_data_df = eval_data_df[["num_time_steps", "average_return"]]
                baseline_cmT_rnT_dfs.append(eval_data_df)
            else:
                print(colored("missing" + dir, "red"))

        baseline_cmT_rnT_df = pd.concat(baseline_cmT_rnT_dfs)
        baseline_cmT_rnT_df = baseline_cmT_rnT_df.groupby(baseline_cmT_rnT_df.index)

        baseline_cmT_rnT_df_mean = baseline_cmT_rnT_df.mean()
        baseline_cmT_rnT_df_sem = baseline_cmT_rnT_df.sem()

        # Compute the mean of the average return at the final time step.
        baseline_cmT_rnT_df_mean = baseline_cmT_rnT_df_mean["average_return"].iloc[-1]

        # Compute the standard error of the mean at the final time step.
        baseline_cmT_rnT_df_sem = baseline_cmT_rnT_df_sem["average_return"].iloc[-1]

        # Compute the confidence interval at the final time step.
        baseline_cmT_rnT_ci = stats.norm.interval(0.95, loc=baseline_cmT_rnT_df_mean, scale=baseline_cmT_rnT_df_sem)
        baseline_cmT_rnT_ci_min = baseline_cmT_rnT_ci[0]
        baseline_cmT_rnT_ci_max = baseline_cmT_rnT_ci[1]

        print("Baseline Confidence Interval: ", baseline_cmT_rnT_ci)

        # cmF_rnF

        cmF_rnF_dfs = []

        for seed in range(0, 30):
            dir = os.path.join(cmF_rnF_path, "seed" + str(seed))
            if os.path.exists(dir):
                eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                eval_data_df = pd.read_csv(eval_data_dir)
                eval_data_df = eval_data_df[["num_time_steps",
                                             "average_return"]]
                cmF_rnF_dfs.append(eval_data_df)
            else:
                print(colored("missing" + dir, "red"))

        cmF_rnF_df = pd.concat(cmF_rnF_dfs)
        cmF_rnF_df = cmF_rnF_df.groupby(cmF_rnF_df.index)

        cmF_rnF_df_mean = cmF_rnF_df.mean()

        for i in range(201, 402):

            mean = cmF_rnF_df_mean["average_return"].iloc[i]
            if baseline_cmT_rnT_ci_min <= mean <= baseline_cmT_rnT_ci_max:
                print("cmF_rnF: ", cmF_rnF_df_mean["num_time_steps"].iloc[i])
                break
            else:
                print("cmF_rnF: ", "----")

        print("done")
