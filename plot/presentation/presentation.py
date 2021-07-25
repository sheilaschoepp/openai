import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from termcolor import colored
from matplotlib.patches import ConnectionPatch

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

    algorithm = None
    ab_env = None
    n_env = None

    ordered_settings = []

    def get_data():
        nonlocal algorithm, ab_env, n_env, ordered_settings

        first_setting = True

        unordered_settings = []

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

            for s in os.listdir(data_dir):

                seed_foldername = os.path.join(data_dir, s)
                csv_filename = os.path.join(seed_foldername, "csv", "eval_data.csv")

                df = pd.read_csv(csv_filename)
                df = df[["num_time_steps", "average_return"]]
                dfs.append(df)

            if len(dfs) != num_seeds:
                # warning to let user know that seeds are missing
                print(colored("The number of seeds for this experiment is 10 but this setting only has {} seeds: {}".format(str(len(dfs)), dir_), "red"))

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
                unordered_settings.append((algorithm, rn, crb, label, df_mean, df_sem))
            elif algorithm == "PPO":
                unordered_settings.append((algorithm, rn, cm, label, df_mean, df_sem))

        assert len(unordered_settings) == 4, "plot_experiment: more than four settings"

        # reorganize settings to obtain a plotting order

        desired_ordering = [(False, False), (False, True), (True, False), (True, True)]  # (rn, crb/cm)

        for do in desired_ordering:
            for s in unordered_settings:
                if do[0] == s[1] and do[1] == s[2]:
                    ordered_settings.append(s)

    get_data()

    # plot settings

    if algorithm == "SAC":
        labels = ["retain all data", "retain network parameters", "retain replay buffer", "retain no data"]
    else:
        labels = ["retain all data", "retain network parameters", "retain memory", "retain no data"]

    plot_directory = os.getcwd() + "/plots"
    os.makedirs(plot_directory, exist_ok=True)

    # 'b' as blue, 'g' as green, 'r' as red, 'c' as cyan, 'm' as magenta, 'y' as yellow, 'k' as black, 'w' as white
    colors = ["b", "g", "m", "k", "r"]

    x_divisor = 1000000

    # create a zoomed subplot

    def plot_zoom():

        fig = plt.figure()

        main = fig.add_subplot(2, 1, 2)
        zoom = fig.add_subplot(2, 6, (2, 5))

        zoom_x_interval_length = 100

        zoom_min_y = -1000
        zoom_max_y = 8000
        zoom_min_x = 20
        zoom_max_x = 25

        x_fault_onset = ordered_settings[0][4].iloc[200, 0] / x_divisor

        # plot normal performance

        fault_end_index = 201

        x = ordered_settings[0][4].iloc[:fault_end_index, 0] / x_divisor
        y = ordered_settings[0][4].iloc[:fault_end_index, 1]

        if ci:
            # 95 % confidence interval
            lb = y - CI_Z * ordered_settings[0][5].iloc[:fault_end_index, 1]
            ub = y + CI_Z * ordered_settings[0][5].iloc[:fault_end_index, 1]
        else:
            # standard error
            lb = y - ordered_settings[0][5].iloc[:fault_end_index, 1]
            ub = y + ordered_settings[0][5].iloc[:fault_end_index, 1]

        main.plot(x, y, color=colors[0], label="normal")
        main.fill_between(x, lb, ub, color=colors[0], alpha=0.3)

        # plot fault performance

        fault_start_index = 200

        for i in range(4):

            x = ordered_settings[i][4].iloc[fault_start_index:, 0] / x_divisor
            y = ordered_settings[i][4].iloc[fault_start_index:, 1]

            if ci:
                # 95 % confidence interval
                lb = y - CI_Z * ordered_settings[i][5].iloc[fault_start_index:, 1]
                ub = y + CI_Z * ordered_settings[i][5].iloc[fault_start_index:, 1]
            else:
                # standard error
                lb = y - ordered_settings[i][5].iloc[fault_start_index:, 1]
                ub = y + ordered_settings[i][5].iloc[fault_start_index:, 1]

            main.plot(x, y, color=colors[i + 1], label=labels[i])
            main.fill_between(x, lb, ub, color=colors[i + 1], alpha=0.3)

            zoom.plot(x, y, color=colors[i + 1], label=labels[i])
            zoom.fill_between(x, lb, ub, color=colors[i + 1], alpha=0.3)

        main.axvline(x=x_fault_onset, color="red")
        main.fill_between((zoom_min_x, zoom_max_x), zoom_min_y, zoom_max_y, facecolor='black', alpha=0.2)

        zoom.axvline(x=20, color="red", lw=4)

        con1 = ConnectionPatch(xyA=(zoom_min_x, zoom_max_y), coordsA=main.transData,
                               xyB=(zoom_min_x, zoom_max_y), coordsB=zoom.transData, color='black')

        main.set_xlim(0, 40)
        zoom.set_xlim(zoom_min_x, zoom_max_x)
        main.set_ylim(-1000, 8000)
        zoom.set_ylim(-1000, 8000)
        main.set_xlabel("million steps")
        main.set_ylabel("average return\n(10 seeds)")
        zoom.set_title(plot_title)
        plt.tight_layout()
        plt.savefig(plot_directory + "/{}_sub.jpg".format(plot_filename))
        plt.show()
        plt.close()

    plot_zoom()

    # plot standard figure

    def plot_all_standard():

        x_fault_onset = ordered_settings[0][4].iloc[200, 0] / x_divisor

        # plot normal performance

        fault_end_index = 201

        x = ordered_settings[0][4].iloc[:fault_end_index, 0] / x_divisor
        y = ordered_settings[0][4].iloc[:fault_end_index, 1]

        if ci:
            # 95 % confidence interval
            lb = y - CI_Z * ordered_settings[0][5].iloc[:fault_end_index, 1]
            ub = y + CI_Z * ordered_settings[0][5].iloc[:fault_end_index, 1]
        else:
            # standard error
            lb = y - ordered_settings[0][5].iloc[:fault_end_index, 1]
            ub = y + ordered_settings[0][5].iloc[:fault_end_index, 1]

        plt.plot(x, y, color=colors[0], label="normal")
        plt.fill_between(x, lb, ub, color=colors[0], alpha=0.3)
        plt.axvline(x=x_fault_onset, color="red")

        # plot fault performance

        fault_start_index = 200

        for i in range(4):

            x = ordered_settings[i][4].iloc[fault_start_index:, 0] / x_divisor
            y = ordered_settings[i][4].iloc[fault_start_index:, 1]

            if ci:
                # 95 % confidence interval
                lb = y - CI_Z * ordered_settings[i][5].iloc[fault_start_index:, 1]
                ub = y + CI_Z * ordered_settings[i][5].iloc[fault_start_index:, 1]
            else:
                # standard error
                lb = y - ordered_settings[i][5].iloc[fault_start_index:, 1]
                ub = y + ordered_settings[i][5].iloc[fault_start_index:, 1]

            plt.plot(x, y, color=colors[i + 1], label=labels[i])
            plt.fill_between(x, lb, ub, color=colors[i + 1], alpha=0.3)

        plt.xlim(0, 40)
        plt.ylim(-1000, 8000)
        plt.xlabel("million steps")
        plt.ylabel("average return (10 seeds)")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig(plot_directory + "/{}_all.jpg".format(plot_filename))
        # plt.show()
        plt.close()

    plot_all_standard()

    # plot one standard

    def plot_one_standard():

        for i in range(4):

            x_fault_onset = ordered_settings[0][4].iloc[200, 0] / x_divisor

            # plot normal performance

            fault_end_index = 201

            x = ordered_settings[0][4].iloc[:fault_end_index, 0] / x_divisor
            y = ordered_settings[0][4].iloc[:fault_end_index, 1]

            if ci:
                # 95 % confidence interval
                lb = y - CI_Z * ordered_settings[0][5].iloc[:fault_end_index, 1]
                ub = y + CI_Z * ordered_settings[0][5].iloc[:fault_end_index, 1]
            else:
                # standard error
                lb = y - ordered_settings[0][5].iloc[:fault_end_index, 1]
                ub = y + ordered_settings[0][5].iloc[:fault_end_index, 1]

            plt.plot(x, y, color=colors[0], label="normal")
            plt.fill_between(x, lb, ub, color=colors[0], alpha=0.3)
            plt.axvline(x=x_fault_onset, color="red")

            # plot fault performance

            fault_start_index = 200

            subscript = ""
            rn = ordered_settings[i][1]
            crb = ordered_settings[i][2]
            if rn:
                subscript = subscript + "_rn"
            if crb:
                subscript = subscript + "_crb"

            x = ordered_settings[i][4].iloc[fault_start_index:, 0] / x_divisor
            y = ordered_settings[i][4].iloc[fault_start_index:, 1]

            if ci:
                # 95 % confidence interval
                lb = y - CI_Z * ordered_settings[i][5].iloc[fault_start_index:, 1]
                ub = y + CI_Z * ordered_settings[i][5].iloc[fault_start_index:, 1]
            else:
                # standard error
                lb = y - ordered_settings[i][5].iloc[fault_start_index:, 1]
                ub = y + ordered_settings[i][5].iloc[fault_start_index:, 1]

            plt.plot(x, y, color=colors[i + 1], label=labels[i])
            plt.fill_between(x, lb, ub, color=colors[i + 1], alpha=0.3)

            plt.xlim(0, 40)
            plt.ylim(-1000, 8000)
            plt.xlabel("million steps")
            plt.ylabel("average return (10 seeds)")
            plt.legend(loc=0)
            plt.tight_layout()
            plt.savefig(plot_directory + "/{}{}.jpg".format(plot_filename, subscript))
            # plt.show()
            plt.close()

    plot_one_standard()


if __name__ == "__main__":

    # number of seeds to plot
    num_seeds = 10

    # confidence interval z value for 9 degrees of freedom (10 seeds)
    if num_seeds == 10:
        CI_Z = 2.262
    else:
        print(colored("__main__: you have specified {} seeds; you must set a new value for CI_Z".format(num_seeds), "red"))
        exit()

    # if True, plot 95% confidence interval; if False, plot standard error
    ci = False

    # plot_experiment("/mnt/DATA/shared/fetchreach/faulty/sac/v1")
    plot_experiment("/Users/sheilaschoepp/Documents/DATA/shared/ant/faulty/sac/v1", "ant_sac_v1", "Soft Actor-Critic (SAC)")
