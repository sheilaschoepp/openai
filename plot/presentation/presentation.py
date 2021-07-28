import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from termcolor import colored
from matplotlib.patches import ConnectionPatch

sns.set_theme()
sns.set_palette("colorblind", color_codes=True)


def plot_experiment(directory):
    """
    Plot all four settings for a single fault.

    @param directory: string
        absolute path for directory containing all experiments for a single fault (e.g. .../shared/fetchreach/faulty/sac/v1/)
    """

    algorithm = ""
    ab_env = ""
    n_env = ""

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

            x = os.listdir(data_dir)
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

            if not rn and not mem:
                label = "retain all data"
            elif not rn and mem:
                label = "retain network parameters"
            elif rn and not mem:
                label = "retain {}".format(storage_type)
            else:  # rn and mem
                label = "retain no data"

            if algorithm == "SAC":
                unordered_settings.append((algorithm, rn, mem, label, df_mean, df_sem))
            elif algorithm == "PPO":
                unordered_settings.append((algorithm, rn, mem, label, df_mean, df_sem))

        assert len(unordered_settings) == 4, "plot_experiment: more than four settings"

        # reorganize settings to obtain a plotting order

        desired_ordering = [(False, False), (False, True), (True, False), (True, True)]  # (rn, mem)

        for do in desired_ordering:
            for s in unordered_settings:
                if do[0] == s[1] and do[1] == s[2]:
                    ordered_settings.append(s)

        print(desired_ordering)
        for i in ordered_settings:
            print(i[0:4])

    get_data()

    # plot settings

    if algorithm == "SAC":
        title = "Soft Actor-Critic (SAC)"
        labels = ["retain all data", "retain network params", "retain replay buffer", "retain no data"]
    else:
        title = "Proximal Policy Optimization (PPO)"
        labels = ["retain all data", "retain network params", "retain memory", "retain no data"]

    plot_directory = os.path.join(os.getcwd(), "plots", env_name, algorithm, ab_env)
    os.makedirs(plot_directory, exist_ok=True)

    # "b" as blue, "g" as green, "r" as red, "c" as cyan, "m" as magenta, "y" as yellow, "k" as black, "w" as white
    colors = ["b", "g", "m", "k", "r"]

    x_divisor = 1000000

    # create a zoomed subplot

    def plot_zoom():

        fig = plt.figure()

        main = fig.add_subplot(2, 1, 1)
        zoom = fig.add_subplot(2, 6, (9, 12))

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

            zoom.plot(x, y, color=colors[i + 1])
            zoom.fill_between(x, lb, ub, color=colors[i + 1], alpha=0.3)

        main.axvline(x=x_fault_onset, color="red")
        main.fill_between((zoom_xmin, zoom_xmax), zoom_ymin, zoom_ymax, facecolor="black", alpha=0.2)

        zoom.axvline(x=zoom_xmin, color="red", lw=4)

        connector1 = ConnectionPatch(xyA=(zoom_xmin, zoom_ymin), coordsA=main.transData,
                                     xyB=(zoom_xmin, zoom_ymax), coordsB=zoom.transData,
                                     color="black",
                                     alpha=0.3)
        fig.add_artist(connector1)

        connector2 = ConnectionPatch(xyA=(zoom_xmax, zoom_ymin), coordsA=main.transData,
                                     xyB=(zoom_xmax, zoom_ymax), coordsB=zoom.transData,
                                     color="black",
                                     alpha=0.3)
        fig.add_artist(connector2)

        fig.legend(bbox_to_anchor=[0.2, 0.25], loc="center")

        main.set_xlim(xmin, xmax)
        zoom.set_xlim(zoom_xmin, zoom_xmax)
        main.set_ylim(ymin, ymax)
        zoom.set_ylim(zoom_ymin, zoom_ymax)
        main.set_xlabel("million steps")
        main.set_ylabel("average return\n(10 seeds)")
        main.set_title(title)
        plt.tight_layout()
        plt.savefig(plot_directory + "/{}_{}_sub.jpg".format(algorithm, ab_env))
        # plt.show()
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

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xlabel("million steps")
        plt.ylabel("average return (10 seeds)")
        plt.legend(bbox_to_anchor=[0.465, 0.35], loc=0)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_directory + "/{}_{}_all.jpg".format(algorithm, ab_env))
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
            mem = ordered_settings[i][2]
            if rn:
                subscript = subscript + "_rn"
            if algorithm == "PPO" and mem:
                subscript = subscript + "_cm"
            elif algorithm == "SAC" and mem:
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

            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.xlabel("million steps")
            plt.ylabel("average return (10 seeds)")
            plt.legend(loc=0)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(plot_directory + "/{}_{}{}.jpg".format(algorithm, ab_env, subscript))
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
    
    """ant"""

    # global for Ant
    env_name = "ant"

    # global ymin/ymax for Ant
    ymin = -1000
    ymax = 8000

    # PPO

    # local for Ant PPO
    xmin = 0
    xmax = 1200

    # local for Ant PPO
    ppo_data_dir = "/media/sschoepp/easystore/shared/ant/faulty/ppo"

    # v1

    zoom_xmin = 600
    zoom_xmax = 660
    zoom_ymin = ymin
    zoom_ymax = ymax

    plot_experiment(os.path.join(ppo_data_dir, "v1"))

    # v2

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    # plot_experiment(os.path.join(ppo_data_dir, "v2"))

    # v3

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    # plot_experiment(os.path.join(ppo_data_dir, "v3"))

    # v4

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    # plot_experiment(os.path.join(ppo_data_dir, "v4"))

    # SAC

    # local for Ant SAC
    xmin = 0
    xmax = 40

    # local for Ant SAC
    sac_data_dir = "/media/sschoepp/easystore/shared/ant/faulty/sac"

    # v1

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v1"))

    # v2

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    # plot_experiment(os.path.join(sac_data_dir, "v2"))

    # v3

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v3"))

    # v4

    zoom_xmin = 20
    zoom_xmax = 25
    zoom_ymin = ymin
    zoom_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v4"))

    """fetchreach"""

    # global for FetchReach
    env_name = "fetchreach"

    # global ymin/ymax for FetchReach
    ymin = -40
    ymax = 5

    # PPO

    # local for FetchReach PPO
    xmin = 0
    xmax = 12

    # local for FetchReach PPO
    ppo_data_dir = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo"

    # v1

    zoom_xmin = 6
    zoom_xmax = 6.5
    zoom_ymin = -12
    zoom_ymax = 1

    plot_experiment(os.path.join(ppo_data_dir, "v1"))

    # v1GE

    zoom_xmin = 6
    zoom_xmax = 6.5
    zoom_ymin = -12
    zoom_ymax = 1

    plot_experiment(os.path.join(ppo_data_dir, "v1GE"))

    # v4

    zoom_xmin = 6
    zoom_xmax = 6.6
    zoom_ymin = -25
    zoom_ymax = 1

    plot_experiment(os.path.join(ppo_data_dir, "v4"))

    # SAC

    # local for FetchReach SAC
    xmin = 0
    xmax = 4

    # local for FetchReach SAC
    sac_data_dir = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac"

    # v1

    zoom_xmin = 2
    zoom_xmax = 2.05
    zoom_ymin = -12
    zoom_ymax = 1

    plot_experiment(os.path.join(sac_data_dir, "v1"))

    # v1GE

    zoom_xmin = 2
    zoom_xmax = 2.05
    zoom_ymin = -15
    zoom_ymax = 1

    plot_experiment(os.path.join(sac_data_dir, "v1GE"))

    # v4

    zoom_xmin = 2
    zoom_xmax = 2.2
    zoom_ymin = -25
    zoom_ymax = 1

    plot_experiment(os.path.join(sac_data_dir, "v4"))
