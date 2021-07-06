import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from termcolor import colored

sns.set_theme()

parser = argparse.ArgumentParser(description="HPS Arguments")

parser.add_argument("-p", "--plot_hps", default=False, action="store_true",
                    help="if True, plot each individual hyperparameter setting individually (default: False)")

parser.add_argument("-n", "--num_best", type=int, default=10, metavar="N",
                    help="plot n best hyperparameter settings (default: 10)")

args = parser.parse_args()


def get_ppo_data(directory):
    """
    Obtain the average return for 200 policy evaluations (or an entire run) across 10 runs for a single hyperparameter setting.

    @param directory: string
        directory for PPO experiment data

    Example: PPOv2_Ant-v2:1000000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu
    """

    parameters = directory.split("_")

    pss = None
    for p in parameters:
        if p.startswith("pss:"):
            pss = int(p.split(":")[1])

    dfs = []

    for i in range(RUNS):

        seed_foldername = directory + "/seed{}".format(i)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]].set_index("num_time_steps").rename(columns={"average_return": str(pss)})
            dfs.append(df)

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        return df_mean, df_sem


def get_ppo_summary_data(directory):
    """
    Obtain summary data for a single hyperparameter setting, averaged across 10 runs.

    Summary data includes:
    - performance mean: average of the (average) return in the last 20 policy evaluations
    - confidence interval lower: lower bound for 95% confidence interval
    - confidence interval upper: upper bound for 95% confidence interval
    - performance mean sum: sum of the average return across 200 policy evaluations (entire run)

    @param directory: string
        directory for PPO experiment data

    Example: PPOv2_Ant-v2:1000000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu
    """

    parameters = directory.split("_")

    pss = None
    for p in parameters:
        if p.startswith("pss:"):
            pss = int(p.split(":")[1])

    dfs = []

    for i in range(RUNS):

        seed_foldername = directory + "/seed{}".format(i)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]].set_index("num_time_steps")
            dfs.append(df)

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_std = df.std()
        df_sem = df.sem()

        performance_mean = df_mean.iloc[-20:, 0].mean()
        performance_std = df_std.iloc[-20:, 0].mean()
        performance_sem = df_sem.iloc[-20:, 0].mean()

        performance_mean_sum = df_mean.iloc[:, 0].sum()

        # confidence interval calculation: 9 degrees of freedom, 95% confidence (or alpha=0.025), table value is 2.262
        # https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        return df_mean, df_sem, [pss, performance_mean, performance_mean - CI_Z * performance_sem, performance_mean + CI_Z * performance_sem, performance_mean_sum]


def get_sac_data(directory):
    """
    Obtain the average return for 200 policy evaluations (or an entire run) across 10 runs for a single hyperparameter setting.

    @param directory: string
        directory for SAC experiment data

    Example: SACv2_Ant-v2:5000000_g:0.91_t:0.0292_a:0.2_lr:0.00425_hd:256_rbs:10000_bs:64_mups:1_tui:1_tef:10000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:3_r
    """

    parameters = directory.split("_")

    pss = None
    for p in parameters:
        if p.startswith("pss:"):
            pss = int(p.split(":")[1])

    dfs = []

    for i in range(RUNS):

        seed_foldername = directory + "/seed{}".format(i)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]].set_index("num_time_steps").rename(columns={"average_return": str(pss)})
            dfs.append(df)

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        return df_mean, df_sem


def get_sac_summary_data(directory):
    """
    Obtain summary data for a single hyperparameter setting, averaged across 10 runs.

    Summary data includes:
    - performance mean: average of the (average) return in the last 20 policy evaluations
    - confidence interval lower: lower bound for 95% confidence interval
    - confidence interval upper: upper bound for 95% confidence interval
    - performance mean sum: sum of the average return across 200 policy evaluations (entire run)

    @param directory: string
        directory for SAC experiment data

    Example: SACv2_Ant-v2:5000000_g:0.91_t:0.0292_a:0.2_lr:0.00425_hd:256_rbs:10000_bs:64_mups:1_tui:1_tef:10000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:3_r
    """

    parameters = directory.split("_")

    pss = None
    for p in parameters:
        if p.startswith("pss:"):
            pss = int(p.split(":")[1])

    dfs = []

    for i in range(RUNS):

        seed_foldername = directory + "/seed{}".format(i)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]].set_index("num_time_steps")
            dfs.append(df)

    if len(dfs) > 0:
        
        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_std = df.std()
        df_sem = df.sem()

        performance_mean = df_mean.iloc[-20:, 0].mean()
        performance_std = df_std.iloc[-20:, 0].mean()
        performance_sem = df_sem.iloc[-20:, 0].mean()

        performance_mean_sum = df_mean.iloc[:, 0].sum()

        # confidence interval calculation: 9 degrees of freedom, 95% confidence (or alpha=0.025), table value is 2.262
        # https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        return df_mean, df_sem, [pss, performance_mean, performance_mean - CI_Z * performance_sem, performance_mean + CI_Z * performance_sem, performance_mean_sum]


def plot_ant_hps(directory, algorithm, df_mean, df_sem):
    """
    Plot hyperparameter settings for the Ant-v2 environment.

    @param directory: string
        data directory
    @param algorithm: string
        algorithm name
    @param df_mean: df
        average return for one parameter setting
    @param df_sem: df
        standard error for one parameter setting
    """

    parameters = directory.split("_")

    pss = None
    for p in parameters:
        if p.startswith("pss:"):
            pss = int(p.split(":")[1])

    plot_directory = os.getcwd() + "/plots/Ant-v2/hps/{}".format(algorithm.upper())
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -2000  # min for y axis
    ymax = 8000  # max for y axis

    x = df_mean.index

    y = df_mean.iloc[:, 0]
    lb = y - CI_Z * df_sem.iloc[:, 0]
    ub = y + CI_Z * df_sem.iloc[:, 0]

    plt.plot(x, y, color="tab:blue")
    plt.fill_between(x, lb, ub, color="tab:blue", alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("Ant-v2 {} pss:{}".format(algorithm.upper(), pss), fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_directory + "/Ant-v2_{}_pss:{}.jpg".format(algorithm.upper(), pss))
    plt.close()


def plot_ant_top(algorithm, seeds, df_means, df_sems):
    """
    Plot best hyperparameter settings for the Ant-v2 environment.

    @param algorithm: string
        algorithm name
    @param seeds: list of int
        list of five best performing seeds, ordered from best performing to least-best performing
    @param df_means: list of df
        list of the average returns for each parameter setting
    @param df_sems: list of df
        list of standard errors for each parameter setting
    """

    plot_directory = os.getcwd() + "/plots/Ant-v2/best/{}".format(algorithm.upper())
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -1000  # min for y axis
    ymax = 8000  # max for y axis

    df_means = pd.concat(df_means, axis=1)
    df_sems = pd.concat(df_sems, axis=1)

    x = df_means.index

    indices = np.arange(args.num_best)

    for i in np.flip(indices):
        column = str(int(seeds[i]))
        y = df_means[column]
        lb = y - CI_Z * df_sems[column]
        ub = y + CI_Z * df_sems[column]
        color = COLORS[i]
        plt.plot(x, y, color=color, label=column)
        plt.fill_between(x, lb, ub, color=color, alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("Ant-v2 {} HPS".format(algorithm.upper()), fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_directory + "/Ant-v2_{}_hps_plot_{}.jpg".format(algorithm.upper(), args.num_best))
    plt.close()


def plot_fetchreach_hps(directory, algorithm, df_mean, df_sem):
    """
    Plot hyperparameter settings for the FetchReach-v1 environment.

    @param directory: string
        data directory
    @param algorithm: string
        algorithm name
    @param df_mean: df
        average return for one parameter setting
    @param df_sem: df
        standard error for one parameter setting
    """

    parameters = directory.split("_")

    pss = None
    for p in parameters:
        if p.startswith("pss:"):
            pss = int(p.split(":")[1])

    plot_directory = os.getcwd() + "/plots/FetchReach-v1/hps/{}".format(algorithm.upper())
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -40  # min for y axis
    ymax = 5  # max for y axis

    x = df_mean.index

    y = df_mean.iloc[:, 0]
    lb = y - CI_Z * df_sem.iloc[:, 0]
    ub = y + CI_Z * df_sem.iloc[:, 0]

    plt.plot(x, y, color="tab:blue")
    plt.fill_between(x, lb, ub, color="tab:blue", alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("FetchReach-v1 {} pss:{}".format(algorithm.upper(), pss), fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_directory + "/FetchReach-v1_{}_pss:{}.jpg".format(algorithm.upper(), pss))
    plt.close()


def plot_fetchreach_top(algorithm, seeds, df_means, df_sems):
    """
    Plot best hyperparameter settings for the FetchReach-v1 environment.

    @param algorithm: string
        algorithm name
    @param seeds: list of int
        list of five best performing seeds, ordered from best performing to least-best performing
    @param df_means: list of df
        list of the average returns for each parameter setting
    @param df_sems: list of df
        list of standard errors for each parameter setting
    """

    plot_directory = os.getcwd() + "/plots/FetchReach-v1/best/{}".format(algorithm.upper())
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -27.5  # min for y axis
    ymax = 2.5  # max for y axis

    df_means = pd.concat(df_means, axis=1)
    df_sems = pd.concat(df_sems, axis=1)

    x = df_means.index

    indices = np.arange(args.num_best)

    for i in np.flip(indices):
        column = str(seeds[i])
        y = df_means[column]
        lb = y - CI_Z * df_sems[column]
        ub = y + CI_Z * df_sems[column]
        color = COLORS[i]
        plt.plot(x, y, color=color, label=column)
        plt.fill_between(x, lb, ub, color=color, alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("FetchReach-v1 {} HPS".format(algorithm.upper()), fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_directory + "/FetchReach-v1_{}_hps_plot_{}.jpg".format(algorithm.upper(), args.num_best))
    plt.close()


def ant():

    hps_data_dir = os.getcwd() + "/data/Ant-v2"
    os.makedirs(hps_data_dir, exist_ok=True)

    """
    PPO
    """

    ppo_data_dir = DATA_DIR + "/ant/hps/ppo"
    ppo_data_dirs = os.listdir(ppo_data_dir)
    ppo_data_dirs.remove("resumed")

    ppo_results = []

    for dir_ in ppo_data_dirs:
        df_mean, df_sem, ppo_result = get_ppo_summary_data(ppo_data_dir + "/" + dir_)
        if args.plot_hps:
            plot_ant_hps(dir_, "ppo", df_mean, df_sem)
        ppo_results.append(ppo_result)

    df = pd.DataFrame(data=ppo_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    os.makedirs(os.path.join(hps_data_dir, "PPO"), exist_ok=True)
    df.to_csv(hps_data_dir + "/PPO/Ant-v2_PPO_hps_data.csv", index=False)

    top_ppo_results_mean = []
    top_ppo_results_sem = []

    df = df["ps seed"].head(args.num_best)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        for dir_ in ppo_data_dirs:
            if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
                df_mean, df_sem = get_ppo_data(ppo_data_dir + "/" + dir_)
                top_ppo_results_mean.append(df_mean)
                top_ppo_results_sem.append(df_sem)
            elif "_resumed" in dir_ and "pss:" + str(seed) == dir_[-(12 + len(str(seed))):-8]:
                df_mean, df_sem = get_ppo_data(ppo_data_dir + "/" + dir_)
                top_ppo_results_mean.append(df_mean)
                top_ppo_results_sem.append(df_sem)

    plot_ant_top("ppo", top_hps_seeds, top_ppo_results_mean, top_ppo_results_sem)

    """
    SAC
    """

    sac_data_dir = DATA_DIR + "/ant/hps/sac"
    sac_data_dirs = os.listdir(sac_data_dir)
    sac_data_dirs.remove("resumed")

    sac_results = []

    for dir_ in sac_data_dirs:
        if "resumed" in dir_:
            df_mean, df_sem, sac_result = get_sac_summary_data(sac_data_dir + "/" + dir_)
            if args.plot_hps:
                plot_ant_hps(dir_, "sac", df_mean, df_sem)
            sac_results.append(sac_result)

    df = pd.DataFrame(data=sac_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df = df.replace(np.nan, 9999999)  # set default hp settings to pss 999

    os.makedirs(os.path.join(hps_data_dir, "SAC"), exist_ok=True)
    df.to_csv(hps_data_dir + "/SAC/Ant-v2_SAC_hps_data.csv", index=False)

    top_sac_results_mean = []
    top_sac_results_sem = []

    df = df["ps seed"].head(args.num_best)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        if seed == float(9999999):  # default setting with no pss value
            df_mean, df_sem = get_sac_data(sac_data_dir + "/" + "SACv2_Ant-v2:20000000_g:0.99_t:0.01_a:0.2_lr:0.0003_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_resumed")
            df_mean.columns = [str(9999999)]
            df_sem.columns = [str(9999999)]
            top_sac_results_mean.append(df_mean)
            top_sac_results_sem.append(df_sem)
        else:
            for dir_ in sac_data_dirs:
                if "resumed" in dir_ and "pss:" + str(int(seed)) == dir_[-(10 + len(str(seed))):-8]:
                    df_mean, df_sem = get_sac_data(sac_data_dir + "/" + dir_)
                    top_sac_results_mean.append(df_mean)
                    top_sac_results_sem.append(df_sem)

    plot_ant_top("sac", top_hps_seeds, top_sac_results_mean, top_sac_results_sem)


def fetchreach():

    hps_data_dir = os.getcwd() + "/data/FetchReach-v1"
    os.makedirs(hps_data_dir, exist_ok=True)

    """
    PPO
    """

    ppo_data_dir = DATA_DIR + "/fetchreach/hps/ppo"
    ppo_data_dirs = os.listdir(ppo_data_dir)

    ppo_results = []

    for dir_ in ppo_data_dirs:
        df_mean, df_sem, ppo_result = get_ppo_summary_data(ppo_data_dir + "/" + dir_)
        if args.plot_hps:
            plot_fetchreach_hps(dir_, "ppo", df_mean, df_sem)
        ppo_results.append(ppo_result)

    df = pd.DataFrame(data=ppo_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    os.makedirs(os.path.join(hps_data_dir, "PPO"), exist_ok=True)
    df.to_csv(hps_data_dir + "/PPO/FetchReach-v1_PPO_hps_data.csv", index=False)

    top_ppo_results_mean = []
    top_ppo_results_sem = []

    df = df["ps seed"].head(args.num_best)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        for dir_ in ppo_data_dirs:
            if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
                df_mean, df_sem = get_ppo_data(ppo_data_dir + "/" + dir_)
                top_ppo_results_mean.append(df_mean)
                top_ppo_results_sem.append(df_sem)

    plot_fetchreach_top("ppo", top_hps_seeds, top_ppo_results_mean, top_ppo_results_sem)

    """
    SAC
    """

    sac_data_dir = DATA_DIR + "/fetchreach/hps/sac"
    sac_data_dirs = os.listdir(sac_data_dir)

    sac_results = []

    for dir_ in sac_data_dirs:
        df_mean, df_sem, sac_result = get_sac_summary_data(sac_data_dir + "/" + dir_)
        if args.plot_hps:
            plot_fetchreach_hps(dir_, "sac", df_mean, df_sem)
        sac_results.append(sac_result)

    df = pd.DataFrame(data=sac_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    os.makedirs(os.path.join(hps_data_dir, "SAC"), exist_ok=True)
    df.to_csv(hps_data_dir + "/SAC/FetchReach-v1_SAC_hps_data.csv", index=False)

    top_sac_results_mean = []
    top_sac_results_sem = []

    df = df["ps seed"].head(args.num_best)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        for dir_ in sac_data_dirs:
            if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
                df_mean, df_sem = get_sac_data(sac_data_dir + "/" + dir_)
                top_sac_results_mean.append(df_mean)
                top_sac_results_sem.append(df_sem)

    plot_fetchreach_top("sac", top_hps_seeds, top_sac_results_mean, top_sac_results_sem)


if __name__ == "__main__":

    DATA_DIR = "/mnt/DATA/shared"

    RUNS = 10  # number of runs (seeds) to average across

    assert 1 <= args.num_best <= 10, "args.num_best must have a value between 1 and 10"

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey", "tab:olive", "tab:cyan"]

    CI_Z = 2.262  # z-score for 95% confidence interval with 9 degrees of freedom

    if args.plot_hps:
        print(colored("plotting all hyperparameter settings", "red"))
        print(colored("this may take awhile...", "red"))

    ant()
    fetchreach()