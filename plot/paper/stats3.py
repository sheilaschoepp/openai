import numpy as np
import os
import pandas as pd
import sys

from termcolor import colored

if __name__ == '__main__':

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

        base_path = f"/Users/sheilaschoepp/Documents/openai/data/ant/exps/complete/ppo/v1/PPOv2_AntEnv-v1:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000"

        cmF_rnF_path = f"{base_path}_cm:False_rn:False_d:cpu"
        cmF_rnT_path = f"{base_path}_cm:False_rn:True_d:cpu"
        cmT_rnF_path = f"{base_path}_cm:True_rn:False_d:cpu"
        baseline_cmT_rnT_path = f"{base_path}_cm:True_rn:True_d:cpu"

        baseline_cmT_rnT_dfs = []

        for seed in range(0, 30):
            dir = os.path.join(baseline_cmT_rnT_path, "seed" + str(seed))
            if os.path.exists(dir):
                eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                eval_data_df = pd.read_csv(eval_data_dir)
                baseline_cmT_rnT_dfs.append(eval_data_df)
            else:
                print(colored("missing" + dir, "red"))

        print(2)
        # AntEnv-v2: Hip 4 ROM
        # AntEnv-v3: Ankle 4 ROM
        # AntEnv-v4: Broken, Unsevered Limb