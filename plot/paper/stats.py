import os
import pandas as pd
import sys

from scipy import stats
from termcolor import colored


if __name__ == "__main__":

    with open("../paper/stats/adaptation_speed_analysis.txt", "w") as f:

        sys.stdout = f

        print("---------------------------------------------------------\n")
        print("Adaptation Speed Analysis for Ant-v2. This analysis\n"
              "displays the time step at which the 95% confidence\n"
              "interval of the average return of our baseline at 300k time steps\n"
              "is reached. Methods that fail to reach this confidence\n"
              "interval are represented with a dashed line.\n")
        print("---------------------------------------------------------")

        print("PPO")
        print("---------------------------------------------------------")

        versions = ["v1", "v2", "v3", "v4"]

        for version in versions:

            if version == "v1":
                print("AntEnv-v1: Broken, Severed Limb")
            elif version == "v2":
                print("AntEnv-v2: Hip 4 ROM")
            elif version == "v3":
                print("AntEnv-v3: Ankle 4 ROM")
            elif version == "v4":
                print("AntEnv-v4: Broken, Unsevered Limb")

            base_path = f"{os.getenv('HOME')}/Documents/openai/data/ant/exps/300k/ppo/{version}/PPOv2_AntEnv-{version}:3000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:5000_ee:10_tmsf:5000"

            # cm = clear memory
            # rn = reinitialize networks

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
                    # print(colored("missing" + dir, "red"))
                    pass

            baseline_cmT_rnT_df = pd.concat(baseline_cmT_rnT_dfs)
            baseline_cmT_rnT_df = baseline_cmT_rnT_df.groupby(baseline_cmT_rnT_df.index)

            baseline_cmT_rnT_df_mean = baseline_cmT_rnT_df.mean()
            baseline_cmT_rnT_df_sem = baseline_cmT_rnT_df.sem()

            # Compute the mean of the average return at 300k time steps.
            baseline_cmT_rnT_df_mean = baseline_cmT_rnT_df_mean["average_return"].iloc[261:262]

            # Compute the standard error of the mean at 300k time steps.
            baseline_cmT_rnT_df_sem = baseline_cmT_rnT_df_sem["average_return"].iloc[261:262]

            # Compute the confidence interval at the final time step.
            baseline_cmT_rnT_ci = stats.norm.interval(0.95, loc=baseline_cmT_rnT_df_mean, scale=baseline_cmT_rnT_df_sem)
            baseline_cmT_rnT_ci_min = baseline_cmT_rnT_ci[0]
            baseline_cmT_rnT_ci_max = baseline_cmT_rnT_ci[1]

            print("Baseline Confidence Interval: ", baseline_cmT_rnT_ci)

            # cmT_rnT

            cmT_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(baseline_cmT_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    cmT_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            cmT_rnT_df = pd.concat(cmT_rnT_dfs)
            cmT_rnT_df = cmT_rnT_df.groupby(cmT_rnT_df.index)

            cmT_rnT_df_mean = cmT_rnT_df.mean()

            time_steps = None
            for i in range(201, 262):

                mean = cmT_rnT_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps = cmT_rnT_df_mean['num_time_steps'].iloc[i]
                    time_steps_ = cmT_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"cmT_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 600000000) / (time_steps - 600000000)) * 100)
                    print(f"{percentage_saved}")
                    break

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
                    # print(colored("missing" + dir, "red"))
                    pass

            cmF_rnF_df = pd.concat(cmF_rnF_dfs)
            cmF_rnF_df = cmF_rnF_df.groupby(cmF_rnF_df.index)

            cmF_rnF_df_mean = cmF_rnF_df.mean()

            for i in range(201, 262):

                mean = cmF_rnF_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps_ = cmF_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"cmF_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 600000000) / (time_steps - 600000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # cmT_rnF

            cmT_rnF_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(cmT_rnF_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    cmT_rnF_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            cmT_rnF_df = pd.concat(cmT_rnF_dfs)
            cmT_rnF_df = cmT_rnF_df.groupby(cmT_rnF_df.index)

            cmT_rnF_df_mean = cmT_rnF_df.mean()

            for i in range(201, 262):

                mean = cmT_rnF_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps_ = cmT_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"cmT_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 600000000) / (time_steps - 600000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # cmF_rnT

            cmF_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(cmF_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    cmF_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            cmF_rnT_df = pd.concat(cmF_rnT_dfs)
            cmF_rnT_df = cmF_rnT_df.groupby(cmF_rnT_df.index)

            cmF_rnT_df_mean = cmF_rnT_df.mean()

            for i in range(201, 262):

                mean = cmF_rnT_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps_ = cmF_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"cmF_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 600000000) / (time_steps - 600000000)) * 100)
                    print(f"{percentage_saved}")
                    break
            print("---------------------------------------------------------")

        print("SAC")
        print("---------------------------------------------------------")

        versions = ["v1", "v2", "v3", "v4"]

        for version in versions:

            if version == "v1":
                print("AntEnv-v1: Broken, Severed Limb")
            elif version == "v2":
                print("AntEnv-v2: Hip 4 ROM")
            elif version == "v3":
                print("AntEnv-v3: Ankle 4 ROM")
            elif version == "v4":
                print("AntEnv-v4: Broken, Unsevered Limb")

            base_path = f"{os.getenv('HOME')}/Documents/openai/data/ant/exps/300k/sac/{version}/SACv2_AntEnv-{version}:300000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:10000"

            # crb = clear replay buffer
            # rn = reinitialize networks

            crbF_rnF_path = f"{base_path}_crb:False_rn:False_a:True_d:cuda_mod"
            crbF_rnT_path = f"{base_path}_crb:False_rn:True_a:True_d:cuda_mod"
            crbT_rnF_path = f"{base_path}_crb:True_rn:False_a:True_d:cuda_mod"
            baseline_crbT_rnT_path = f"{base_path}_crb:True_rn:True_a:True_d:cuda_mod"

            # BASELINE crbT_rnT

            baseline_crbT_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(baseline_crbT_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[
                        ["num_time_steps", "average_return"]]
                    baseline_crbT_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            baseline_crbT_rnT_df = pd.concat(baseline_crbT_rnT_dfs)
            baseline_crbT_rnT_df = baseline_crbT_rnT_df.groupby(baseline_crbT_rnT_df.index)

            baseline_crbT_rnT_df_mean = baseline_crbT_rnT_df.mean()
            baseline_crbT_rnT_df_sem = baseline_crbT_rnT_df.sem()

            # Compute the mean of the average return at the final time step.
            baseline_crbT_rnT_df_mean = baseline_crbT_rnT_df_mean["average_return"].iloc[231:232]

            # Compute the standard error of the mean at the final time step.
            baseline_crbT_rnT_df_sem = baseline_crbT_rnT_df_sem["average_return"].iloc[231:232]

            # Compute the confidence interval at the final time step.
            baseline_crbT_rnT_ci = stats.norm.interval(0.95,
                                                      loc=baseline_crbT_rnT_df_mean,
                                                      scale=baseline_crbT_rnT_df_sem)
            baseline_crbT_rnT_ci_min = baseline_crbT_rnT_ci[0]
            baseline_crbT_rnT_ci_max = baseline_crbT_rnT_ci[1]

            print("Baseline Confidence Interval: ", baseline_crbT_rnT_ci)

            # crbT_rnT

            crbT_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(baseline_crbT_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbT_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbT_rnT_df = pd.concat(crbT_rnT_dfs)
            crbT_rnT_df = crbT_rnT_df.groupby(crbT_rnT_df.index)

            crbT_rnT_df_mean = crbT_rnT_df.mean()

            time_steps = None
            for i in range(201, 232):

                mean = crbT_rnT_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps = crbT_rnT_df_mean['num_time_steps'].iloc[i]
                    time_steps_ = crbT_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"crbT_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 20000000) / (time_steps - 20000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # crbF_rnF

            crbF_rnF_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(crbF_rnF_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbF_rnF_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbF_rnF_df = pd.concat(crbF_rnF_dfs)
            crbF_rnF_df = crbF_rnF_df.groupby(crbF_rnF_df.index)

            crbF_rnF_df_mean = crbF_rnF_df.mean()

            for i in range(201, 232):

                mean = crbF_rnF_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps_ = crbF_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"crbF_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 20000000) / (time_steps - 20000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # crbT_rnF

            crbT_rnF_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(crbT_rnF_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbT_rnF_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbT_rnF_df = pd.concat(crbT_rnF_dfs)
            crbT_rnF_df = crbT_rnF_df.groupby(crbT_rnF_df.index)

            crbT_rnF_df_mean = crbT_rnF_df.mean()

            for i in range(201, 232):

                mean = crbT_rnF_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps_ = crbT_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"crbT_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 20000000) / (time_steps - 20000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # crbF_rnT

            crbF_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(crbF_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbF_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbF_rnT_df = pd.concat(crbF_rnT_dfs)
            crbF_rnT_df = crbF_rnT_df.groupby(crbF_rnT_df.index)

            crbF_rnT_df_mean = crbF_rnT_df.mean()

            for i in range(201, 232):

                mean = crbF_rnT_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps_ = crbF_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"crbF_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 20000000) / (time_steps - 20000000)) * 100)
                    print(f"{percentage_saved}")
                    break
            print("---------------------------------------------------------")

        """FetchReach"""

        print("PPO")
        print("---------------------------------------------------------")

        versions = ["v4", "v6"]

        for version in versions:

            if version == "v4":
                print("FetchReachEnv-v4: Frozen Shoulder Lift Position Sensor")
            elif version == "v6":
                print("FetchReachEnv-v6: Elbow Flex Position Slippage")

            base_path = f"{os.getenv('HOME')}/Documents/openai/data/fetchreach/exps/300k/ppo/{version}/PPO_FetchReachEnv-{version}:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_slrd:1.0_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000"

            # cm = clear memory
            # rn = reinitialize networks

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
                    # print(colored("missing" + dir, "red"))
                    pass

            baseline_cmT_rnT_df = pd.concat(baseline_cmT_rnT_dfs)
            baseline_cmT_rnT_df = baseline_cmT_rnT_df.groupby(baseline_cmT_rnT_df.index)

            baseline_cmT_rnT_df_mean = baseline_cmT_rnT_df.mean()
            baseline_cmT_rnT_df_sem = baseline_cmT_rnT_df.sem()

            # Compute the mean of the average return at the final time step.
            baseline_cmT_rnT_df_mean = baseline_cmT_rnT_df_mean["average_return"].iloc[231:232]

            # Compute the standard error of the mean at the final time step.
            baseline_cmT_rnT_df_sem = baseline_cmT_rnT_df_sem["average_return"].iloc[231:232]

            # Compute the confidence interval at the final time step.
            baseline_cmT_rnT_ci = stats.norm.interval(0.95, loc=baseline_cmT_rnT_df_mean, scale=baseline_cmT_rnT_df_sem)
            baseline_cmT_rnT_ci_min = baseline_cmT_rnT_ci[0]
            baseline_cmT_rnT_ci_max = baseline_cmT_rnT_ci[1]

            print("Baseline Confidence Interval: ", baseline_cmT_rnT_ci)

            # cmT_rnT

            cmT_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(baseline_cmT_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    cmT_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            cmT_rnT_df = pd.concat(cmT_rnT_dfs)
            cmT_rnT_df = cmT_rnT_df.groupby(cmT_rnT_df.index)

            cmT_rnT_df_mean = cmT_rnT_df.mean()

            time_steps = None
            for i in range(201, 232):

                mean = cmT_rnT_df_mean["average_return"].iloc[i]
                time_steps = cmT_rnT_df_mean['num_time_steps'].iloc[i]
                time_steps_ = cmT_rnT_df_mean['num_time_steps'].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    print(f"cmT_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 6000000) / (time_steps - 6000000)) * 100)
                    print(f"{percentage_saved}")
                    break

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
                    # print(colored("missing" + dir, "red"))
                    pass

            cmF_rnF_df = pd.concat(cmF_rnF_dfs)
            cmF_rnF_df = cmF_rnF_df.groupby(cmF_rnF_df.index)

            cmF_rnF_df_mean = cmF_rnF_df.mean()

            for i in range(201, 232):

                mean = cmF_rnF_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps_ = cmF_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"cmF_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 6000000) / (time_steps - 6000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # cmT_rnF

            cmT_rnF_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(cmT_rnF_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    cmT_rnF_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            cmT_rnF_df = pd.concat(cmT_rnF_dfs)
            cmT_rnF_df = cmT_rnF_df.groupby(cmT_rnF_df.index)

            cmT_rnF_df_mean = cmT_rnF_df.mean()

            for i in range(201, 232):

                mean = cmT_rnF_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps_ = cmT_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"cmT_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 6000000) / (time_steps - 6000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # cmF_rnT

            cmF_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(cmF_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    cmF_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            cmF_rnT_df = pd.concat(cmF_rnT_dfs)
            cmF_rnT_df = cmF_rnT_df.groupby(cmF_rnT_df.index)

            cmF_rnT_df_mean = cmF_rnT_df.mean()

            for i in range(201, 232):

                mean = cmF_rnT_df_mean["average_return"].iloc[i]
                if baseline_cmT_rnT_ci_min <= mean:
                    time_steps_ = cmF_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"cmF_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 6000000) / (time_steps - 6000000)) * 100)
                    print(f"{percentage_saved}")
                    break
            print("---------------------------------------------------------")

        print("SAC")
        print("---------------------------------------------------------")

        versions = ["v4", "v6"]

        for version in versions:

            if version == "v4":
                print("FetchReachEnv-v4: Frozen Shoulder Lift Position Sensor")
            elif version == "v6":
                print("FetchReachEnv-v6: Elbow Flex Position Slippage")

            base_path = f"{os.getenv('HOME')}/Documents/openai/data/fetchreach/exps/300k/sac/{version}/SAC_FetchReachEnv-{version}:300000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000"

            # crb = clear replay buffer
            # rn = reinitialize networks

            crbF_rnF_path = f"{base_path}_crb:False_rn:False_a:True_d:cuda"
            crbF_rnT_path = f"{base_path}_crb:False_rn:True_a:True_d:cuda"
            crbT_rnF_path = f"{base_path}_crb:True_rn:False_a:True_d:cuda"
            baseline_crbT_rnT_path = f"{base_path}_crb:True_rn:True_a:True_d:cuda"

            # BASELINE crbT_rnT

            baseline_crbT_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(baseline_crbT_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[
                        ["num_time_steps", "average_return"]]
                    baseline_crbT_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            baseline_crbT_rnT_df = pd.concat(baseline_crbT_rnT_dfs)
            baseline_crbT_rnT_df = baseline_crbT_rnT_df.groupby(baseline_crbT_rnT_df.index)

            baseline_crbT_rnT_df_mean = baseline_crbT_rnT_df.mean()
            baseline_crbT_rnT_df_sem = baseline_crbT_rnT_df.sem()

            # Compute the mean of the average return at the final time step.
            baseline_crbT_rnT_df_mean = baseline_crbT_rnT_df_mean["average_return"].iloc[231:232]

            # Compute the standard error of the mean at the final time step.
            baseline_crbT_rnT_df_sem = baseline_crbT_rnT_df_sem["average_return"].iloc[231:232]

            # Compute the confidence interval at the final time step.
            baseline_crbT_rnT_ci = stats.norm.interval(0.95,
                                                      loc=baseline_crbT_rnT_df_mean,
                                                      scale=baseline_crbT_rnT_df_sem)
            baseline_crbT_rnT_ci_min = baseline_crbT_rnT_ci[0]
            baseline_crbT_rnT_ci_max = baseline_crbT_rnT_ci[1]

            print("Baseline Confidence Interval: ", baseline_crbT_rnT_ci)

            # crbT_rnT

            crbT_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(baseline_crbT_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbT_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbT_rnT_df = pd.concat(crbT_rnT_dfs)
            crbT_rnT_df = crbT_rnT_df.groupby(crbT_rnT_df.index)

            crbT_rnT_df_mean = crbT_rnT_df.mean()

            time_steps = None
            for i in range(201, 232):

                mean = crbT_rnT_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps = crbT_rnT_df_mean['num_time_steps'].iloc[i]
                    time_steps_ = crbT_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"crbT_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    # print(f"{percentage_saved} = 100 - ((({time_steps_} - 2000000) / ({time_steps} - 2000000)) * 100)")
                    percentage_saved = 100 - (((time_steps_ - 2000000) / (time_steps - 2000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # crbF_rnF

            crbF_rnF_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(crbF_rnF_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv", "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbF_rnF_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbF_rnF_df = pd.concat(crbF_rnF_dfs)
            crbF_rnF_df = crbF_rnF_df.groupby(crbF_rnF_df.index)

            crbF_rnF_df_mean = crbF_rnF_df.mean()

            for i in range(201, 232):

                mean = crbF_rnF_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps_ = crbF_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"crbF_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    # print(f"{percentage_saved} = 100 - ((({time_steps_} - 2000000) / ({time_steps} - 2000000)) * 100)")
                    percentage_saved = 100 - (((time_steps_ - 2000000) / (time_steps - 2000000)) * 100)
                    print(f"{percentage_saved}")
                    break

            # crbT_rnF

            crbT_rnF_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(crbT_rnF_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbT_rnF_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbT_rnF_df = pd.concat(crbT_rnF_dfs)
            crbT_rnF_df = crbT_rnF_df.groupby(crbT_rnF_df.index)

            crbT_rnF_df_mean = crbT_rnF_df.mean()

            for i in range(201, 232):

                mean = crbT_rnF_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps_ = crbT_rnF_df_mean['num_time_steps'].iloc[i]
                    print(f"crbT_rnF: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 2000000) / (time_steps - 2000000)) * 100)
                    # print(f"{percentage_saved} = 100 - ((({time_steps_} - 2000000) / ({time_steps} - 2000000)) * 100)")
                    print(f"{percentage_saved}")
                    break

            # crbF_rnT

            crbF_rnT_dfs = []

            for seed in range(0, 30):
                dir = os.path.join(crbF_rnT_path, "seed" + str(seed))
                if os.path.exists(dir):
                    eval_data_dir = os.path.join(dir, "csv",
                                                 "eval_data.csv")
                    eval_data_df = pd.read_csv(eval_data_dir)
                    eval_data_df = eval_data_df[["num_time_steps",
                                                 "average_return"]]
                    crbF_rnT_dfs.append(eval_data_df)
                else:
                    # print(colored("missing" + dir, "red"))
                    pass

            crbF_rnT_df = pd.concat(crbF_rnT_dfs)
            crbF_rnT_df = crbF_rnT_df.groupby(crbF_rnT_df.index)

            crbF_rnT_df_mean = crbF_rnT_df.mean()

            for i in range(201, 232):

                mean = crbF_rnT_df_mean["average_return"].iloc[i]
                if baseline_crbT_rnT_ci_min <= mean:
                    time_steps_ = crbF_rnT_df_mean['num_time_steps'].iloc[i]
                    print(f"crbF_rnT: "
                          f"t={time_steps_} "
                          f"mean={mean}")
                    percentage_saved = 100 - (((time_steps_ - 2000000) / (time_steps - 2000000)) * 100)
                    # print(f"{percentage_saved} = 100 - ((({time_steps_} - 2000000) / ({time_steps} - 2000000)) * 100)")
                    print(f"{percentage_saved}")
                    break
            print("---------------------------------------------------------")
        print("done")