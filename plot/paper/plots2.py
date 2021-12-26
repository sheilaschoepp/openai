import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import ConnectionPatch
from termcolor import colored
from PIL import Image

sns.set_theme()
# sns.set_palette("colorblind", color_codes=True)
# print(sns.color_palette("colorblind").as_hex())
# [blue, orange, green, red, purple, brown, pink, grey, green, aqua]
# ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#949494", "#ece133", "#56b4e9"]
# converted above colour pallet to WACG 2.0 compliant colors using https://webaim.org/resources/contrastchecker/
#                   blue       orange      green       red      purple      brown      pink       grey       green      aqua
# palette_colours = ["#0173b2", "#875603", "#027957", "#AD4B00", "#A63F93", "#915C30", "#C3098E", "#696969", "#70680A", "#156FA2"]
#                   blue        green  magneta/purple  black     red
palette_colours = ["#0173b2", "#027957", "#A63F93", "#000000", "#AD4B00"]

LARGE = 16
MEDIUM = 14

plt.rc("axes", titlesize=LARGE)     # fontsize of the axes title
plt.rc("axes", labelsize=LARGE)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM)   # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM)   # fontsize of the tick labels
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"


def plot_experiment(directory):
    """
    Plot all four settings for a single fault.

    @param directory: string
        absolute path for directory containing all experiments for a single fault (e.g. .../data/fetchreach/exps/sac/v1/)
    """

    # get data

    algorithm = ""
    ab_env = ""
    n_env = ""

    ordered_settings = []

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

        cs = None
        rn = None
        for p in parameters:
            if "crb:" in p or "cm:" in p:
                cs = eval(p.split(":")[1])
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

        if len(dfs) < num_seeds:
            # warning to let user know that seeds are missing
            print(colored("The number of seeds for this experiment is {} but this setting only has {} seeds: {}".format(num_seeds, str(len(dfs)), dir_), "red"))

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        if algorithm == "SAC":
            unordered_settings.append((algorithm, rn, cs, df_mean, df_sem))
        elif algorithm == "PPO":
            unordered_settings.append((algorithm, rn, cs, df_mean, df_sem))

    assert len(unordered_settings) == 4, "plot_experiment: not four settings"

    # reorganize settings to obtain a plotting order
    # (rn, cs) desired_ordering = [(False, False), (False, True), (True, False), (True, True)]

    ordered_settings = unordered_settings.copy()
    ordered_settings.sort()

    # plot

    eval_fault_onset = 201
    eval_fault_stop = 235

    if algorithm == "SAC":
        title = "Soft Actor-Critic (SAC)"
    else:
        title = "Proximal Policy Optimization (PPO)"

    plot_directory = os.path.join(os.getcwd(), "plots", env_name, algorithm, ab_env)
    os.makedirs(plot_directory, exist_ok=True)

    x_divisor = 1000000

    x_fault_onset = ordered_settings[0][3].iloc[eval_fault_onset, 0] / x_divisor

    if algorithm == "PPO":
        x_fault_onset -= 600
    elif algorithm == "SAC":
        x_fault_onset -= 20

    # plot normal performance

    normal = ordered_settings[0][3].iloc[eval_fault_onset - 10:eval_fault_onset, 0] / x_divisor

    # plt.plot(x, y, color=palette_colours[0], label="normal")
    # plt.fill_between(x, lb, ub, color=palette_colours[0], alpha=0.3)
    # plt.axvline(x=x_fault_onset, color="red", ymin=0.95)

    # plot fault performance

    for i in range(4):

        x = ordered_settings[i][3].iloc[eval_fault_onset:eval_fault_stop, 0] / x_divisor
        y = ordered_settings[i][3].iloc[eval_fault_onset:eval_fault_stop, 1]

        # 95 % confidence interval
        lb = y - CI_Z * ordered_settings[i][4].iloc[eval_fault_onset:eval_fault_stop, 1]
        ub = y + CI_Z * ordered_settings[i][4].iloc[eval_fault_onset:eval_fault_stop, 1]

        plt.plot(x, y, color=palette_colours[i + 1])
        plt.fill_between(x, lb, ub, color=palette_colours[i + 1], alpha=0.2)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("million steps")
    plt.ylabel("average return ({} seeds)".format(num_seeds))
    # plt.legend(bbox_to_anchor=[0.465, 0.35], loc=0)
    plt.title(title)
    plt.tight_layout()
    filename = plot_directory + "/{}_{}_all_mod.jpg".format(algorithm, ab_env)
    plt.savefig(filename, dpi=300)
    # Image.open(filename).convert("CMYK").save(filename)
    plt.show()
    plt.close()


def legend():

    fig, ax = plt.subplots()
    ax.axis("off")

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", palette_colours[i]) for i in range(5)]
    labels = ["pre-fault\ntask", "retain NN params,\nretain storage", "retain NN params,\ndiscard storage", "discard NN params,\nretain storage", "discard NN params,\ndiscard storage"]
    legend = plt.legend(handles, labels, ncol=5, loc=1, framealpha=1, frameon=False)

    def export_legend(legend, filename="legend.jpg"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        filename = "plots/{}".format(filename)
        fig.savefig(filename, dpi=300, bbox_inches=bbox)
        # # Image.open(filename).convert("CMYK").save(filename)

    export_legend(legend)
    plt.close()


def legend2():

    fig, ax = plt.subplots()
    ax.axis("off")

    e = lambda: plt.plot([], [], ls="--", color=palette_colours[0])[0]
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles_ = [e()]
    handles = [f("s", palette_colours[i]) for i in range(1, 5)]
    handles.insert(0, handles_[0])
    labels = ["pre-fault\ntask", "retain NN params,\nretain storage", "retain NN params,\ndiscard storage", "discard NN params,\nretain storage", "discard NN params,\ndiscard storage"]
    legend = plt.legend(handles, labels, ncol=5, loc=1, framealpha=1, frameon=False)

    def export_legend(legend, filename="legend2.jpg"):
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        filename = "plots/{}".format(filename)
        fig.savefig(filename, dpi=300, bbox_inches=bbox)
        # # Image.open(filename).convert("CMYK").save(filename)

    export_legend(legend)
    plt.close()


if __name__ == "__main__":

    legend()
    legend2()

    PROJECT_PATH = pathlib.Path(os.getcwd()).parents[1]
    DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, "data")

    # number of seeds to plot
    num_seeds = 30

    # 95% confidence interval z value (99% confidence interval z value = 2.576)
    CI_Z = 1.960

    # if True, plot 95% confidence interval; if False, plot standard error
    ci = True

    """ant"""

    # global for Ant
    env_name = "ant"

    # global ymin/ymax for Ant
    ymin = -1000
    ymax = 8000

    # PPO

    # local for Ant PPO
    xmin = 600
    xmax = 700

    # local for Ant PPO
    ppo_data_dir = DATA_FOLDER_PATH + "/ant/exps/ppo"

    # v1

    zoom_init_xmin = 600
    zoom_init_xmax = 660
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(ppo_data_dir, "v1"))

    # v2

    zoom_init_xmin = 600
    zoom_init_xmax = 660
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(ppo_data_dir, "v2"))

    # v3

    zoom_init_xmin = 600
    zoom_init_xmax = 660
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(ppo_data_dir, "v3"))

    # v4

    zoom_init_xmin = 600
    zoom_init_xmax = 660
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(ppo_data_dir, "v4"))

    # SAC

    # local for Ant SAC
    xmin = 0
    xmax = 40

    # local for Ant SAC
    sac_data_dir = DATA_FOLDER_PATH + "/ant/exps/sac"

    # v1

    zoom_init_xmin = 20
    zoom_init_xmax = 22
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v1"))

    # v2

    zoom_init_xmin = 20
    zoom_init_xmax = 22
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v2"))

    # v3

    zoom_init_xmin = 20
    zoom_init_xmax = 22
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v3"))

    # v4

    zoom_init_xmin = 20
    zoom_init_xmax = 22
    zoom_init_ymin = ymin
    zoom_init_ymax = ymax

    plot_experiment(os.path.join(sac_data_dir, "v4"))

    """fetchreach"""

    # global for FetchReach
    env_name = "fetchreach"

    # global ymin/ymax for FetchReach
    ymin = -30
    ymax = 5

    # PPO

    # local for FetchReach PPO
    xmin = 0
    xmax = 12

    # local for FetchReach PPO
    ppo_data_dir = DATA_FOLDER_PATH + "/fetchreach/exps/ppo"

    # v4

    zoom_init_xmin = 6
    zoom_init_xmax = 6.25
    zoom_init_ymin = -25
    zoom_init_ymax = 1

    plot_experiment(os.path.join(ppo_data_dir, "v4"))

    # v6

    zoom_init_xmin = 6
    zoom_init_xmax = 6.25
    zoom_init_ymin = -25
    zoom_init_ymax = 1

    plot_experiment(os.path.join(ppo_data_dir, "v6"))

    # SAC

    # local for FetchReach SAC
    xmin = 0
    xmax = 4

    # local for FetchReach SAC
    sac_data_dir = DATA_FOLDER_PATH + "/fetchreach/exps/sac"

    # v4

    zoom_init_xmin = 2
    zoom_init_xmax = 2.1
    zoom_init_ymin = -25
    zoom_init_ymax = 1

    plot_experiment(os.path.join(sac_data_dir, "v4"))

    # v6

    zoom_init_xmin = 2
    zoom_init_xmax = 2.1
    zoom_init_ymin = -25
    zoom_init_ymax = 1

    plot_experiment(os.path.join(sac_data_dir, "v6"))