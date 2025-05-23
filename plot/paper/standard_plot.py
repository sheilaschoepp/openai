import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from termcolor import colored
from PIL import Image

sns.set_theme()
# #                    blue       black      orange      green     pink
# palette_colours = ["#0173b2", "#000000", "#AD4B00", "#027957", "#A63F93"]
#                    blue       orange      green      red     purple
palette_colours = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]

LARGE = 16
MEDIUM = 14

plt.rc("axes", titlesize=LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM)  # fontsize of the tick labels
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"


def plot_early_adaptation(directory):
    """
    Plot early adaptation for all four settings of a single fault.

    @param directory: string
        absolute path for directory containing all experiments for a single fault (e.g. .../data/fetchreach/exps/sac/v1/)
    """

    # get data

    algorithm = ""
    ab_env = ""
    n_env = ""
    ts_fault_onset = None

    first_setting = True

    ordered_settings = []

    for dir_ in os.listdir(directory):

        # dir_ has format: SACv2_FetchReachEnvGE-v1:2000000_FetchReachEnvGE-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda
        parameters = dir_.split("_")

        # obtain setting

        if first_setting:
            algorithm = parameters[0][:-2]
            ab_env = parameters[1].split(":")[0]
            n_env = parameters[2].split(":")[0]
            ts_fault_onset = int(parameters[2].split(":")[1])
            first_setting = False
        else:
            # check to make sure that the four setting folders contained within directory are for the same algorithm, normal environment, and abnormal environment
            setting_algorithm = parameters[0][:-2]
            setting_ab_env = parameters[1].split(":")[0]
            setting_n_env = parameters[2].split(":")[0]
            setting_ts_fault_onset = int(parameters[2].split(":")[1])
            assert setting_algorithm == algorithm, "plot_experiment: folders in directory are for more than one algorithm"
            assert setting_ab_env == ab_env, "plot_experiment: folders in directory are for more than one abnormal environment"
            assert setting_n_env == n_env, "plot_experiment: folders in directory are for more than one normal environment"
            assert setting_ts_fault_onset == ts_fault_onset, "plot_experiment: folders in directory are for more than one normal environment"

        if parameters[-1] == "r":
            # skip over resumable experiments (final episode is not complete)
            continue

        cs = None
        rn = None
        for p in parameters:
            if "crb:" in p or "cm:" in p:
                cs = eval(p.split(":")[1])
            elif "rn:" in p:
                rn = eval(p.split(":")[1])

        label = None
        if not cs and not rn:
            label = "retain NN params\nretain storage"
        elif cs and not rn:
            label = "retain NN params\ndiscard storage"
        elif not cs and rn:
            label = "discard NN params\nretain storage"
        elif cs and rn:
            label = "discard NN params\ndiscard storage"

        # obtain data for setting: mean and standard error

        dfs = []

        data_dir = os.path.join(directory, dir_)

        for s in os.listdir(data_dir):
            seed_foldername = os.path.join(data_dir, s)
            csv_filename = os.path.join(seed_foldername, "csv", "eval_data.csv")

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]]
            dfs.append(df)

        if len(dfs) < num_seeds:
            # warning to let user know that seeds are missing
            print(colored(
                "The number of seeds for this experiment is {} but this setting only has {} seeds: {}".format(
                    num_seeds, str(len(dfs)), dir_), "red"))

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        # mean = df['average_return'].iloc[-1]
        # sem = df['average_return'].sem()

        ordered_settings.append(
            (algorithm, rn, cs, label, df_mean, df_sem))  # TODO

    assert len(ordered_settings) == 4, "plot_experiment: not four settings"

    # reorganize settings to obtain a plotting order
    # (rn, cs) desired_ordering = [(True, True), (True, False), (False, True), (False, False)]

    ordered_settings.sort(reverse=False)

    # plot

    eval_fault_onset = 201

    x_divisor = 1000000

    x_fault_onset = ordered_settings[0][4].iloc[eval_fault_onset, 0] - ts_fault_onset

    for i in range(4):
        # # asymptotic performance
        # x_asymp = ordered_settings[i][4].iloc[eval_fault_onset:, 0] - ts_fault_onset
        # y_asymp = ordered_settings[i][4].iloc[-10:, 1].mean()
        #
        # # cut plots when asymptotic performance is reached
        # y_array = ordered_settings[i][4].iloc[eval_fault_onset:, 1].to_numpy()
        eval_fault_stop = 402
        # for j in range(len(y_array)):
        #     if y_array[j] >= y_asymp:
        #         eval_fault_stop = j + 1 + eval_fault_onset
        #         break

        # data
        x = (ordered_settings[i][4].iloc[eval_fault_onset:eval_fault_stop,
             0] - ts_fault_onset) / x_divisor
        y = ordered_settings[i][4].iloc[eval_fault_onset:eval_fault_stop, 1]

        # 95 % confidence interval
        lb = y - CI_Z * ordered_settings[i][5].iloc[
                        eval_fault_onset:eval_fault_stop, 1]
        ub = y + CI_Z * ordered_settings[i][5].iloc[
                        eval_fault_onset:eval_fault_stop, 1]

        label_ = ordered_settings[i][3]

        plt.plot(x, y, color=palette_colours[i + 1], label=label_)
        plt.fill_between(x, lb, ub, color=palette_colours[i + 1], alpha=0.2)
        # plt.axhline(y=y_asymp, color=palette_colours[i + 1], linestyle="dashed", linewidth=1)

    # plt.axvline(x=x_fault_onset, color="red", ymin=0.95, linewidth=4)
    plt.xlim(xmin - (ts_fault_onset / x_divisor),
             xmax - (ts_fault_onset / x_divisor))
    plt.ylim(ymin, ymax)
    plt.xlabel("million steps")
    plt.ylabel("average return")

    if algorithm == "SAC":
        title = "Soft Actor-Critic"
    else:
        title = "Proximal Policy Optimization"

    plt.title(title)

    # plt.legend()

    plt.tight_layout()

    plot_directory = os.path.join(os.getcwd(), "plots", env_name, algorithm,
                                  "standard_plot", ab_env)
    os.makedirs(plot_directory, exist_ok=True)
    filename = plot_directory + "/{}_{}_all_mod.jpg".format(algorithm, ab_env)
    plt.savefig(filename, dpi=300)
    # Image.open(filename).convert("CMYK").save(filename)
    # plt.show()
    plt.close()


if __name__ == "__main__":

    data_folder_path = f"{os.getenv('HOME')}/Documents/openai/data"

    # number of seeds to plot
    num_seeds = 30

    # 95% confidence interval z value (99% confidence interval z value = 2.576)
    CI_Z = 1.960

    """ant"""

    # global for Ant
    env_name = "ant"

    # global ymin/ymax for Ant
    ymin = -1000
    ymax = 8000

    """ant PPO"""

    # local for Ant PPO
    ppo_data_dir = data_folder_path + "/ant/exps/complete/ppo"

    # local for Ant PPO
    xmin = 600
    xmax = 660

    # v1

    plot_early_adaptation(os.path.join(ppo_data_dir, "v1"))

    # v2

    plot_early_adaptation(os.path.join(ppo_data_dir, "v2"))

    # v3

    plot_early_adaptation(os.path.join(ppo_data_dir, "v3"))

    # v4

    plot_early_adaptation(os.path.join(ppo_data_dir, "v4"))

    """ant SAC"""

    # local for Ant SAC
    sac_data_dir = data_folder_path + "/ant/exps/complete/sac"

    # local for Ant SAC
    xmin = 20
    xmax = 22

    # v1

    plot_early_adaptation(os.path.join(sac_data_dir, "v1"))

    # v2

    plot_early_adaptation(os.path.join(sac_data_dir, "v2"))

    # v3

    plot_early_adaptation(os.path.join(sac_data_dir, "v3"))

    # v4

    plot_early_adaptation(os.path.join(sac_data_dir, "v4"))

    """fetchreach"""

    # global for FetchReach
    env_name = "fetchreach"

    # global ymin/ymax for FetchReach
    ymin = -30
    ymax = 5

    # PPO

    # local for FetchReach PPO
    ppo_data_dir = data_folder_path + "/fetchreach/exps/complete/ppo"

    # local for FetchReach PPO
    xmin = 6
    xmax = 6.6

    # v4

    plot_early_adaptation(os.path.join(ppo_data_dir, "v4"))

    # v6

    plot_early_adaptation(os.path.join(ppo_data_dir, "v6"))

    # SAC

    # local for FetchReach SAC
    sac_data_dir = data_folder_path + "/fetchreach/exps/complete/sac"

    # local for FetchReach SAC
    xmin = 2
    xmax = 2.2

    # v4

    plot_early_adaptation(os.path.join(sac_data_dir, "v4"))

    # v6

    plot_early_adaptation(os.path.join(sac_data_dir, "v6"))
