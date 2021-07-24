import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from termcolor import colored

sns.set_theme()
sns.set_palette("colorblind", color_codes=True)


def plot_experiment(directory, plot_filename, plot_title=""):
    """
    Plot all four settings for a single fault.

    @param directory: string
        absolute path for directory containing all experiments for a single fault (e.g. .../shared/fetchreach/faulty/sac/v1/)
    @param plot_filename: string
        filename for the plot
    @param plot_title: string
        title for the plot
    """

    first_setting = True

    algorithm = None
    ab_env = None
    n_env = None

    settings = []

    for dir_ in os.listdir(directory):

        # dir_ has format: SACv2_FetchReachEnvGE-v1:2000000_FetchReachEnvGE-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda
        parameters = dir_.split("_")

        # obtain setting

        if first_setting:
            algorithm = parameters[0][:-2]
            ab_env = parameters[1].split(":")[0]
            n_env = parameters[2].split(":")[0]
            first_setting = False
        else:
            # check to make sure that the four setting folders contained within directory are for the same algorithm, normal environment, and abnormal environment
            setting_algorithm = parameters[0][:-2]
            setting_ab_env = parameters[1].split(":")[0]
            setting_n_env = parameters[2].split(":")[0]
            assert setting_algorithm == algorithm, "plot_experiment: folders in directory are for more than one algorithm"
            assert setting_ab_env == ab_env, "plot_experiment: folders in directory are for more than one abnormal environment"
            assert setting_n_env == n_env, "plot_experiment: folders in directory are for more than one normal environment"

        if parameters[-1] == "r":
            # skip over resumable experiments (final episode is not complete)
            continue

        crb = None
        cm = None
        rn = None
        for p in parameters:
            if "crb:" in p:
                crb = eval(p.split(":")[1])
            elif "cm:" in p:
                cm = eval(p.split(":")[1])
            elif "rn:" in p:
                rn = eval(p.split(":")[1])

        # obtain data for setting: mean and standard error

        dfs = []

        data_dir = os.path.join(directory, dir_)
        print(algorithm, crb, rn)
        for s in os.listdir(data_dir):

            seed_foldername = os.path.join(data_dir, s)
            csv_filename = os.path.join(seed_foldername, "csv", "eval_data.csv")

            df = pd.read_csv(csv_filename)

            if df.iloc[200, -1] < df.iloc[199, -1]:
                # before fault application: add the runtime entry after resuming to the last runtime entry before resuming
                # assumption: experiment was resumed to finish a single episode
                df.iloc[200, -1] += df.iloc[199, -1]

            if df.iloc[-1, -1] < df.iloc[-2, -1]:
                # after fault application: add the runtime entry after resuming to the last runtime entry before resuming
                # assumption: experiment was resumed to finish a single episode
                df.iloc[-1, -1] += df.iloc[-2, -1]

            # add the last runtime entry before application of fault to all entries after application of fault
            df.iloc[201:, -1:] += df.iloc[200, -1]

            df = df[["num_time_steps", "real_time", "average_return"]]
            dfs.append(df)

        if len(dfs) < num_seeds:
            # warning to let user know that seeds are missing
            print(colored(
                "The number of seeds for this experiment setting is less than 10 and is equal to {}: {}".format(
                    str(len(dfs)), dir_), "red"))

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        # label
        if algorithm == "SAC":
            storage_type = "replay buffer"
        else:
            storage_type = "memory"

        if not rn and not crb:
            label = "retain all data"
        elif not rn and crb:
            label = "retain network parameters"
        elif rn and not crb:
            label = "retain {}".format(storage_type)
        else:  # rn and crb
            label = "retain no data"

        if algorithm == "SAC":
            settings.append((algorithm, rn, crb, df_mean, df_sem, label))
        elif algorithm == "PPO":
            settings.append((algorithm, rn, cm, df_mean, df_sem, label))

    assert len(settings) == 4, "plot_experiment: more than four settings"

    # reorganize settings to obtain a plotting order

    ordered_settings = []  # [[0 algorithm, 1 rn, 2 crb, 3 df_mean, 4 df_sem]]

    desired_ordering = [(False, False), (False, True), (True, False), (True, True)]  # (rn, crb/cm)

    for do in desired_ordering:
        for s in settings:
            if do[0] == s[1] and do[1] == s[2]:
                ordered_settings.append(s)
    ordered_settings = settings

    # labels

    if algorithm == "SAC":
        labels = ["retain all data", "retain network parameters", "retain replay buffer", "retain no data"]
    else:
        labels = ["retain all data", "retain network parameters", "retain memory", "retain no data"]

    # plot settings

    plot_directory = os.getcwd() + "/plots"
    os.makedirs(plot_directory, exist_ok=True)

    # 'b' as blue, 'g' as green, 'r' as red, 'c' as cyan, 'm' as magenta, 'y' as yellow, 'k' as black, 'w' as white
    colors = ["b", "g", "m", "k", "r"]

    time_unit = "days"
    time_divisor = 60 * 60 * 24  # convert seconds to days

    _, ax = plt.subplots()

    x_fault_onset = ordered_settings[0][3].iloc[200, 1] / time_divisor

    # plot normal performance

    x = ordered_settings[0][3].iloc[:201, 1] / time_divisor
    y = ordered_settings[0][3].iloc[:201, 2]

    if ci:
        # 95 % confidence interval
        lb = y - CI_Z * ordered_settings[0][4].iloc[:201, 2]  # ordered_settings[0][4]: df_sem
        ub = y + CI_Z * ordered_settings[0][4].iloc[:201, 2]
    else:
        # standard error
        lb = y - ordered_settings[0][4].iloc[:201, 2]  # ordered_settings[0][4]: df_sem
        ub = y + ordered_settings[0][4].iloc[:201, 2]

    ax.plot(x, y, color=colors[0], label="normal")
    ax.fill_between(x, lb, ub, color=colors[0], alpha=0.3)
    plt.axvline(x=x_fault_onset, color="red")

    # plot fault performance

    for i in range(4):

        x = ordered_settings[i][3].iloc[201:, 1] / time_divisor  # ordered_settings[0][3]: df_mean
        y = ordered_settings[i][3].iloc[201:, 2]

        # 95 % confidence interval
        lb = y - CI_Z * ordered_settings[i][4].iloc[201:, 2]  # ordered_settings[0][4]: df_sem
        ub = y + CI_Z * ordered_settings[i][4].iloc[201:, 2]

        # standard error
        lb = y - ordered_settings[i][4].iloc[201:, 2]  # ordered_settings[0][4]: df_sem
        ub = y + ordered_settings[i][4].iloc[201:, 2]

        ax.plot(x, y, color=colors[i + 1], label=labels[i])
        ax.fill_between(x, lb, ub, color=colors[i + 1], alpha=0.3)

    plt.ylim(-1500, 8000)
    plt.xlabel("real time ({})".format(time_unit))
    plt.ylabel("average return (10 seeds)")
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_directory + "/{}.jpg".format(plot_filename))
    plt.show()
    plt.close()


if __name__ == "__main__":

    # number of seeds to plot
    num_seeds = 10

    # confidence interval z value for 9 degrees of freedom (10 seeds)
    if num_seeds == 10:
        CI_Z = 2.262

    # if True, plot 95% confidence interval; if False, plot standard error
    ci = False

    # plot_experiment("/mnt/DATA/shared/fetchreach/faulty/sac/v1")
    plot_experiment("/mnt/DATA/shared/ant/faulty/sac/v1", "ant_sac_v1", "Soft Actor-Critic (SAC)")
