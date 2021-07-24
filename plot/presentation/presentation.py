import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

sns.set_theme()
sns.color_palette("dark")


def plot_experiment(directory):
    """
    Plot all four settings for a single fault.

    @param directory: string
        absolute path for directory containing all experiments for a single fault (e.g. .../shared/fetchreach/faulty/sac/v1/)
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

        # warning to let user know that seeds are missing
        if len(dfs) < num_seeds:

            print(colored("The number of seeds for this experiment setting is less than 10", "red"))

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        if algorithm == "SAC":
            settings.append((algorithm, crb, rn, df_mean, df_sem))
        elif algorithm == "PPO":
            settings.append((algorithm, cm, rn, df_mean, df_sem))

    assert len(settings) == 4, "plot_experiment: more than four settings"

    # plot settings

    plot_directory = os.getcwd() + "/plots"
    os.makedirs(plot_directory, exist_ok=True)

    colors = ["blue", "olive", "grey", "brown", "purple", "pink", "red", "cyan", "orange", "green"]

    _, ax = plt.subplots()

    for i in range(4):

        x = settings[i][3].iloc[201:, 1]
        y = settings[i][3].iloc[201:, 2]

        # 95 % confidence interval
        lb = y - CI_Z * settings[i][4].iloc[201:, 2]
        ub = y + CI_Z * settings[i][4].iloc[201:, 2]

        # standard error
        lb = y - settings[i][4].iloc[201:, 2]
        ub = y + settings[i][4].iloc[201:, 2]

        label = None
        if algorithm == "SAC":
            if not settings[i][1] and not settings[i][2]:
                label = "normal"
            elif not settings[i][1] and settings[i][2]:
                label = "crb"
            elif settings[i][1] and not settings[i][2]:
                label = "rn"
            elif settings[i][1] and settings[i][2]:
                label = "crb and rn"
        elif algorithm == "PPO":
            pass  # todo

        ax.plot(x, y, color=colors[i+1], label=label)
        ax.fill_between(x, lb, ub, color=colors[i+1], alpha=0.3)

    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

    plt.xlabel("time steps")
    # plt.ylim(ymin, ymax)
    # plt.ylabel("average\nreturn\n({} seeds)".format(num_seeds), labelpad=35).set_rotation(0)
    # plt.title("{}".format(title), fontweight="bold")
    # plt.tight_layout()
    # plt.savefig(plot_directory + "/{}.jpg".format(title))
    # # plt.show()
    # plt.close()


if __name__ == "__main__":

    num_seeds = 10

    # confidence interval z value for 9 degrees of freedom (10 seeds)
    if num_seeds == 10:
        CI_Z = 2.262

    # plot_experiment("/mnt/DATA/shared/fetchreach/faulty/sac/v1")
    plot_experiment("/mnt/DATA/shared/ant/faulty/sac/v1")
