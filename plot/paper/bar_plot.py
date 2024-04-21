import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats as st
import seaborn as sns
import sys

from PIL import Image
from scipy.stats import sem, ttest_ind
from termcolor import colored

"""
Plot Settings
"""

# Set the default theme for seaborn plots to the library's theme.
sns.set_theme()

# Define a custom color palette with specific hexadecimal color codes
# for blue, orange, green, red, and purple.
palette_colours = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]

# Set constants for large and medium font sizes.
LARGE = 16
MEDIUM = 14

# Set the font size of titles on axes to 'LARGE'.
plt.rc("axes", titlesize=LARGE)     # Font size of the axes title.
plt.rc("axes", labelsize=LARGE)     # Font size of the x and y labels.

# Set the font size of ticks on axes to 'MEDIUM'.
plt.rc("xtick", labelsize=MEDIUM)   # Font size of the x tick labels.
plt.rc("ytick", labelsize=MEDIUM)   # Font size of the y tick labels.

# Set the font type for PDF exports to Type 42 (also known as TrueType).
plt.rcParams['pdf.fonttype'] = 42

# Set the font type for PostScript exports to Type 42 (TrueType).
plt.rcParams['ps.fonttype'] = 42

# Set the default font family for all text in plots to "Times New Roman".
plt.rcParams["font.family"] = "Times New Roman"

"""
Functions
"""


def convert_to_text(number):
    return '{:,}'.format(number)


def compute_post_performances(directory, fault_time_steps, performances_data):

    # experiment info
    info = directory.split("/")[-1].split("_")

    algo = None
    nts = None  # normal time steps
    env = None
    rn = None  # reinitialize networks
    cs = None  # clear storage

    for entry in info:
        if entry.startswith("SAC") or entry.startswith("PPO"):
            algo = entry
        elif entry.startswith("Ant-") or entry.startswith("FetchReachEnv-v0"):
            nts = eval(entry.split(":")[1])
        elif (entry.startswith("AntEnv") or entry.startswith("FetchReachEnv")) and not entry.startswith("FetchReachEnv-v0"):
            env = entry.split(":")[0]
        elif entry.startswith("rn:"):
            rn = eval(entry.split(":")[1])
        elif entry.startswith("crb:") or entry.startswith("cm:"):
            cs = eval(entry.split(":")[1])

    post = []

    for seed in range(0, 30):
        dir1 = os.path.join(directory, "seed" + str(seed))
        if os.path.exists(dir1):
            eval_data_dir = os.path.join(dir1, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            indexes = eval_data.loc[eval_data["num_time_steps"] == nts+fault_time_steps].index
            post.append(eval_data.loc[indexes[-1], "average_return"])
        else:
            print(colored("missing" + dir1, "red"))

    post = np.array(post).flatten()

    post_mean = np.mean(post)

    post_std = np.std(post, ddof=1)
    post_sem = post_std / np.sqrt(len(post))

    ci_z = 1.960

    if env.startswith("Ant"):
        post_mean = round(post_mean)
        post_ci = round(post_sem * ci_z)
    elif env.startswith("FetchReach"):
        post_mean = round(post_mean, 3)
        post_ci = round(post_sem * ci_z, 3)

    performances_data.append([algo, env, rn, cs, post_mean, post_ci])


def plot_bar_plots():

    algorithms = ["PPO", "SAC"]
    environments = ["Ant", "FetchReach"]

    for algorithm in algorithms:

        for environment in environments:

            data = []

            rnFcsF = None
            rnFcsT = None
            rnTcsF = None
            rnTcsT = None
            rnFcsF_ci = None
            rnFcsT_ci = None
            rnTcsF_ci = None
            rnTcsT_ci = None

            # list of list: entries [algo, env, rn, cs, mean, ci]

            for entry in partial_post_performances:
                algo = entry[0]
                env = entry[1]
                if algo.startswith(algorithm) and env.startswith(environment):
                    rn = entry[2]
                    cs = entry[3]
                    post_mean = entry[4]
                    post_ci = entry[5]

                    # list of list: entries [algo, env, rn, cs, mean, ci]
                    for entry_ in complete_post_performances:
                        if entry_[0].startswith(algo) and entry_[1].startswith(env) and entry_[2] == True and entry_[3] == True:
                            post_asym = entry_[4]

                    if not rn and not cs:
                        rnFcsF = post_mean
                        rnFcsF_ci = post_ci
                    elif not rn and cs:
                        rnFcsT = post_mean
                        rnFcsT_ci = post_ci
                    elif rn and not cs:
                        rnTcsF = post_mean
                        rnTcsF_ci = post_ci
                    elif rn and cs:
                        rnTcsT = post_mean
                        rnTcsT_ci = post_ci
                    if rnFcsF is not None and rnFcsT is not None and rnTcsF is not None and rnTcsT is not None:
                        data.append([algo,
                                     env,
                                     rnFcsF,
                                     rnFcsT,
                                     rnTcsF,
                                     rnTcsT,
                                     rnFcsF_ci,
                                     rnFcsT_ci,
                                     rnTcsF_ci,
                                     rnTcsT_ci,
                                     post_asym])

                        rnFcsF = None
                        rnFcsT = None
                        rnTcsF = None
                        rnTcsT = None
                        rnFcsF_ci = None
                        rnFcsT_ci = None
                        rnTcsF_ci = None
                        rnTcsT_ci = None

            # plot
            if environment == "Ant":

                ts = convert_to_text(ant_fault_time_steps)

                # sort data by env name
                data.sort()
                # reorganize data to
                # ["AntEnv-v2", "AntEnv-v3", "AntEnv-v1", "AntEnv-v4"]
                data_ = [data[1], data[2], data[0], data[3]]
                labels = ["hip ROM\nrestriction",
                          "ankle ROM\nrestriction",
                          "broken,\nsevered limb",
                          "broken,\nunsevered limb"]

                # reorganize the data
                rnFcsFs = []
                rnFcsTs = []
                rnTcsFs = []
                rnTcsTs = []
                rnFcsFs_sem = []
                rnFcsTs_sem = []
                rnTcsFs_sem = []
                rnTcsTs_sem = []
                asymp_mean_baselines = []
                for entry in data_:
                    rnFcsFs.append(entry[2])
                    rnFcsTs.append(entry[3])
                    rnTcsFs.append(entry[4])
                    rnTcsTs.append(entry[5])
                    rnFcsFs_sem.append(entry[6])
                    rnFcsTs_sem.append(entry[7])
                    rnTcsFs_sem.append(entry[8])
                    rnTcsTs_sem.append(entry[9])
                    asymp_mean_baselines.append(entry[10])

                bar_width = 0.2
                br1 = np.arange(len(labels))
                br2 = [x + bar_width for x in br1]
                br3 = [x + bar_width for x in br2]
                br4 = [x + bar_width for x in br3]

                plt.bar(br1, np.array(rnFcsFs), yerr=rnFcsFs_sem, color=palette_colours[1], width=bar_width)
                plt.bar(br2, np.array(rnFcsTs), yerr=rnFcsTs_sem, color=palette_colours[2], width=bar_width)
                plt.bar(br3, np.array(rnTcsFs), yerr=rnTcsFs_sem, color=palette_colours[3], width=bar_width)
                plt.bar(br4, np.array(rnTcsTs), yerr=rnTcsTs_sem, color=palette_colours[4], width=bar_width)

                plt.axhline(xmin=0.05, xmax=0.23, y=asymp_mean_baselines[0], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.29, xmax=0.47, y=asymp_mean_baselines[1], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.53, xmax=0.71, y=asymp_mean_baselines[2], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.77, xmax=0.95, y=asymp_mean_baselines[3], color=palette_colours[4], linestyle="dashed", linewidth=1)

                plt.axvline(x=0.3, color="black", ymax=0.025)
                plt.axvline(x=1.3, color="black", ymax=0.025)
                plt.axvline(x=2.3, color="black", ymax=0.025)
                plt.axvline(x=3.3, color="black", ymax=0.025)

                plt.xticks([r + 0.3 for r in range(len(labels))],  labels)
                plt.yticks([-2000, -1000, 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000])

            elif environment == "FetchReach":

                ts = convert_to_text(fetchreach_fault_time_steps)

                data.sort()
                data_ = data
                #                            v4                                  v6
                labels = ["frozen shoulder lift\nposition sensor", "elbow flex\nposition slippage"]

                # reorganize the data
                # labels = []
                rnFcsFs = []
                rnFcsTs = []
                rnTcsFs = []
                rnTcsTs = []
                rnFcsFs_sem = []
                rnFcsTs_sem = []
                rnTcsFs_sem = []
                rnTcsTs_sem = []
                asymp_mean_baselines = []
                for entry in data_:
                    # labels.append(entry[1])
                    rnFcsFs.append(entry[2])
                    rnFcsTs.append(entry[3])
                    rnTcsFs.append(entry[4])
                    rnTcsTs.append(entry[5])
                    rnFcsFs_sem.append(entry[6])
                    rnFcsTs_sem.append(entry[7])
                    rnTcsFs_sem.append(entry[8])
                    rnTcsTs_sem.append(entry[9])
                    asymp_mean_baselines.append(entry[10])

                bar_width = 0.2
                br1 = np.arange(len(labels))
                br2 = [x + bar_width for x in br1]
                br3 = [x + bar_width for x in br2]
                br4 = [x + bar_width for x in br3]

                plt.bar(br1, np.array(rnFcsFs), yerr=rnFcsFs_sem, color=palette_colours[1], width=bar_width, bottom=asymp_mean_baselines)
                plt.bar(br2, np.array(rnFcsTs), yerr=rnFcsTs_sem, color=palette_colours[2], width=bar_width, bottom=asymp_mean_baselines)
                plt.bar(br3, np.array(rnTcsFs), yerr=rnTcsFs_sem, color=palette_colours[3], width=bar_width, bottom=asymp_mean_baselines)
                plt.bar(br4, np.array(rnTcsTs), yerr=rnTcsTs_sem, color=palette_colours[4], width=bar_width, bottom=asymp_mean_baselines)

                plt.axhline(xmin=0.05, xmax=0.45, y=asymp_mean_baselines[0], color=palette_colours[4], linestyle="dashed", linewidth=1)
                plt.axhline(xmin=0.555, xmax=0.955, y=asymp_mean_baselines[1], color=palette_colours[4], linestyle="dashed", linewidth=1)

                plt.axvline(x=0.3, color="black", ymax=0.025)
                plt.axvline(x=1.3, color="black", ymax=0.025)

                plt.xticks([r + 0.3 for r in range(len(labels))], labels)
                plt.yticks(np.arange(-30, 1, 5))
                plt.ylim((-30, 1.5))

            if algorithm.startswith("PPO"):
                plt.title("Proximal Policy Optimization: {} Time Steps".format(ts))
            elif algorithm.startswith("SAC"):
                plt.title("Soft Actor-Critic: {} Time Steps".format(ts))

            plt.xlabel("fault")
            plt.ylabel("average return (30 seeds)")
            plt.tight_layout()

            plot_directory = os.path.join(os.getcwd(), "plots", environment.lower(), algorithm)
            os.makedirs(plot_directory, exist_ok=True)

            filename = plot_directory + "/{}_{}_average_return_after_fault_onset_{}.jpg".format(algorithm.upper(), environment, ts)
            plt.savefig(filename, dpi=300)
            # Image.open(filename).convert("CMYK").save(filename)

            # plt.show()

            plt.close()


def legend():
    fig, ax = plt.subplots(figsize=(3, 0.24))
    ax.axis("off")

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", palette_colours[i]) for i in range(1,5)]
    theta = r"$\theta$"
    emmm = r"$\mathcal{M}$"
    labels = [f"retain {theta},\nretain {emmm}",
              f"retain {theta},\ndiscard {emmm}",
              f"discard {theta},\nretain {emmm}",
              f"discard {theta},\ndiscard {emmm}"]
    legend = plt.legend(handles, labels, loc="center", ncol=4, framealpha=1,
                        frameon=True, facecolor="inherit", prop={"size": 6})
    fig.canvas.draw()

    # Save the figure with the legend, without padding.
    filename = "plots/{}".format("legend.jpg")
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def legend2():
    fig, ax = plt.subplots(figsize=(4, 0.24))
    ax.axis("off")

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", palette_colours[i]) for i in range(5)]
    theta = r"$\theta$"
    emmm = r"$\mathcal{M}$"
    labels = ["pre-fault",
              f"retain {theta},\nretain {emmm}",
              f"retain {theta},\ndiscard {emmm}",
              f"discard {theta},\nretain {emmm}",
              f"discard {theta},\ndiscard {emmm}"]
    legend = plt.legend(handles, labels, loc="center", ncol=5, framealpha=1,
                        frameon=True, facecolor="inherit", prop={"size": 6})
    fig.canvas.draw()

    # Save the figure with the legend, without padding.
    filename = "plots/{}".format("legend2.jpg")
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


"""
Main Function (Entry Point)
"""

if __name__ == "__main__":

    # list of list: entries [algo, env, rn, cs, mean, ci]
    partial_post_performances = []
    complete_post_performances = []

    """Ant"""

    """Complete Performances"""

    # Data directory for Ant at 300k time steps.
    ant_data_dir = f"{os.getenv('HOME')}/Documents/openai/data/ant/exps/complete"

    # For comparison, the number of time steps in the Ant fault
    # environment.
    ant_fault_time_steps = None

    # Cycle through ppo/sac and v1/v2/v3/v4 to compute post-fault
    # performances.
    for algo in os.listdir(ant_data_dir):
        if algo == "ppo":
            ant_fault_time_steps = 600000000
        elif algo == "sac":
            ant_fault_time_steps = 20000000
        algo_dir = os.path.join(ant_data_dir, algo)
        for version in os.listdir(algo_dir):
            version_dir = os.path.join(algo_dir, version)
            for exp in os.listdir(version_dir):
                exp_dir = os.path.join(version_dir, exp)
                compute_post_performances(exp_dir,
                                          ant_fault_time_steps,
                                          complete_post_performances)

    """Partial Performances"""

    # Data directory for Ant at 300k time steps.
    ant_data_dir = f"{os.getenv('HOME')}/Documents/openai/data/ant/exps/300k"

    # For comparison, the number of time steps in the Ant fault
    # environment.
    ant_fault_time_steps = 200000  # todo

    # Cycle through ppo/sac and v1/v2/v3/v4 to compute post-fault
    # performances.
    for algo in os.listdir(ant_data_dir):  # ppo / sac
        algo_dir = os.path.join(ant_data_dir, algo)
        for version in os.listdir(algo_dir):  # v1 / v2 / v3 / v4
            version_dir = os.path.join(algo_dir, version)
            for exp in os.listdir(version_dir):  # cs / rn
                exp_dir = os.path.join(version_dir, exp)
                compute_post_performances(exp_dir,
                                          ant_fault_time_steps,
                                          partial_post_performances)

    """FetchReach"""

    """Complete Performances"""

    # Data directory for FetchReach at 300k time steps.
    fetchreach_data_dir = f"{os.getenv('HOME')}/Documents/openai/data/fetchreach/exps/complete"

    # For comparison, the number of time steps in the FetchReach fault
    # environment.
    fetchreach_fault_time_steps = None

    # Cycle through ppo/sac and v1/v2/v3/v4 to compute post-fault
    # performances.
    for algo in os.listdir(fetchreach_data_dir):
        if algo == "ppo":
            fetchreach_fault_time_steps = 6000000
        elif algo == "sac":
            fetchreach_fault_time_steps = 2000000
        algo_dir = os.path.join(fetchreach_data_dir, algo)
        for version in os.listdir(algo_dir):
            version_dir = os.path.join(algo_dir, version)
            for exp in os.listdir(version_dir):
                exp_dir = os.path.join(version_dir, exp)
                compute_post_performances(exp_dir,
                                          fetchreach_fault_time_steps,
                                          complete_post_performances)

    """Partial Performances"""

    # Data directory for FetchReach at 300k time steps.
    fetchreach_data_dir = f"{os.getenv('HOME')}/Documents/openai/data/fetchreach/exps/300k"

    # For comparison, the number of time steps in the FetchReach fault
    # environment.
    fetchreach_fault_time_steps = 0  # todo

    # Cycle through ppo/sac and v1/v2/v3/v4 to compute post-fault
    # performances.
    for algo in os.listdir(fetchreach_data_dir):  # ppo / sac
        algo_dir = os.path.join(fetchreach_data_dir, algo)
        for version in os.listdir(algo_dir):  # v1 / v2 / v3 / v4
            version_dir = os.path.join(algo_dir, version)
            for exp in os.listdir(version_dir):  # cs / rn
                exp_dir = os.path.join(version_dir, exp)
                compute_post_performances(exp_dir,
                                          fetchreach_fault_time_steps,
                                          partial_post_performances)

    # plot_bar_plots()

    legend()
    legend2()
