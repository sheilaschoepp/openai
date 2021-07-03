import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import utils.plot_style_settings as pss

sns.set_theme()


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


def get_modified_ppo_data(directory):
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
            df = df[["num_time_steps", "average_return"]].rename(columns={"average_return": str(pss)})
            non_zero_rows = (df != 0).any(axis=1)
            df = df.loc[non_zero_rows]
            dfs.append(df)

    min_length = len(dfs[0])
    for df in dfs:
        num_rows = df.shape[0]
        if num_rows < min_length:
            min_length = num_rows

    dfs = [df[:min_length] for df in dfs]

    dfs = [df.set_index("num_time_steps") for df in dfs]

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
            df = df["average_return"]
            dfs.append(df)

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_std = df.std()
        df_sem = df.sem()

        performance_mean = df_mean[-20:].mean()
        performance_std = df_std[-20:].mean()
        performance_sem = df_sem[-20:].mean()

        performance_mean_sum = df_mean.sum()

        # confidence interval calculation: 9 degrees of freedom, 95% confidence (or alpha=0.025), table value is 2.262
        # https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        return [pss, performance_mean, performance_mean - 2.262 * performance_sem, performance_mean + 2.262 * performance_sem, performance_mean_sum]


def get_modified_ppo_summary_data(directory):
    """
    Obtain modified summary data for a single hyperparameter setting, averaged across 10 runs.

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
            df = df["average_return"]
            non_zero_rows = (df != 0)
            df = df.loc[non_zero_rows]
            dfs.append(df)

    min_length = len(dfs[0])
    for df in dfs:
        num_rows = df.shape[0]
        if num_rows < min_length:
            min_length = num_rows

    dfs = [df[:min_length] for df in dfs]

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_std = df.std()
        df_sem = df.sem()

        performance_mean = df_mean[-20:].mean()
        performance_std = df_std[-20:].mean()
        performance_sem = df_sem[-20:].mean()

        performance_mean_sum = df_mean.sum()

        # confidence interval calculation: 9 degrees of freedom, 95% confidence (or alpha=0.025), table value is 2.262
        # https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        return [pss, performance_mean, performance_mean - 2.262 * performance_sem, performance_mean + 2.262 * performance_sem, performance_mean_sum]


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
            df = df["average_return"]
            dfs.append(df)

    if len(dfs) > 0:
        
        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_std = df.std()
        df_sem = df.sem()

        performance_mean = df_mean[-20:].mean()
        performance_std = df_std[-20:].mean()
        performance_sem = df_sem[-20:].mean()

        performance_mean_sum = df_mean.sum()

        # confidence interval calculation: 9 degrees of freedom, 95% confidence (or alpha=0.025), table value is 2.262
        # https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        return [pss, performance_mean, performance_mean - 2.262*performance_sem, performance_mean + 2.262*performance_sem, performance_mean_sum]


def plot_modified_ant(algorithm, seeds, df_means, df_sems):
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

    plot_directory = os.getcwd() + "/plots/Ant-v2"
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -1000  # min for y axis
    ymax = 8000  # max for y axis

    # df_means = pd.concat(df_means, axis=1)
    # df_sems = pd.concat(df_sems, axis=1)

    x = df_means[0].reset_index()["num_time_steps"][:163]

    indices = np.arange(NUM_BEST)

    for i in np.flip(indices):
        column = str(seeds[i])
        y = df_means[i].reset_index()[str(seeds[i])][:163]
        lb = y - df_sems[i].reset_index()[str(seeds[i])][:163]
        ub = y + df_sems[i].reset_index()[str(seeds[i])][:163]
        color = COLORS[i]
        plt.plot(x, y, color=color, label=column)
        plt.fill_between(x, lb, ub, color=color, alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("Ant-v2 {} HPS".format(algorithm.upper()), fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_directory + "/Ant-v2_{}_hps_plot_{}.jpg".format(algorithm.upper(), NUM_BEST))
    plt.show()
    plt.close()


def plot_ant(algorithm, seeds, df_means, df_sems):
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

    plot_directory = os.getcwd() + "/plots/Ant-v2"
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -1000  # min for y axis
    ymax = 8000  # max for y axis

    df_means = pd.concat(df_means, axis=1)
    df_sems = pd.concat(df_sems, axis=1)

    x = df_means.index

    indices = np.arange(NUM_BEST)

    for i in np.flip(indices):
        column = str(seeds[i])
        y = df_means[str(seeds[i])]
        lb = y - df_sems[str(seeds[i])]
        ub = y + df_sems[str(seeds[i])]
        color = COLORS[i]
        plt.plot(x, y, color=color, label=column)
        plt.fill_between(x, lb, ub, color=color, alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("Ant-v2 {} HPS".format(algorithm.upper()), fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_directory + "/Ant-v2_{}_hps_plot.jpg".format(algorithm.upper()))
    plt.show()
    plt.close()


def plot_fetchreach(algorithm, seeds, df_means, df_sems):
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

    plot_directory = os.getcwd() + "/plots/FetchReach-v1"
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -27.5  # min for y axis
    ymax = 2.5  # max for y axis

    df_means = pd.concat(df_means, axis=1)
    df_sems = pd.concat(df_sems, axis=1)

    x = df_means.index

    indices = np.arange(NUM_BEST)

    for i in np.flip(indices):
        column = str(seeds[i])
        y = df_means[str(seeds[i])]
        lb = y - df_sems[str(seeds[i])]
        ub = y + df_sems[str(seeds[i])]
        color = COLORS[i]
        plt.plot(x, y, color=color, label=column)
        plt.fill_between(x, lb, ub, color=color, alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(10 seeds)", labelpad=35).set_rotation(0)
    plt.title("FetchReach-v1 {} HPS".format(algorithm.upper()), fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_directory + "/FetchReach-v1_{}_hps_plot.jpg".format(algorithm.upper()))
    plt.close()


def ant():

    hps_data_dir = os.getcwd() + "/data/Ant-v2"
    os.makedirs(hps_data_dir, exist_ok=True)

    """
    PPO
    """

    ppo_data_dir = DATA_DIR + "/ant/hpsc/ppo"
    ppo_data_dirs = os.listdir(ppo_data_dir)

    ppo_results = []

    for dir_ in ppo_data_dirs:
        ppo_result = get_modified_ppo_summary_data(ppo_data_dir + "/" + dir_)
        ppo_results.append(ppo_result)

    df = pd.DataFrame(data=ppo_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/Ant-v2_PPO_hps_data_{}.csv".format(NUM_BEST), index=False)

    top_ppo_results_mean = []
    top_ppo_results_sem = []

    df = df["ps seed"].head(NUM_BEST)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        for dir_ in ppo_data_dirs:
            if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
                df_mean, df_sem = get_modified_ppo_data(ppo_data_dir + "/" + dir_)
                top_ppo_results_mean.append(df_mean)
                top_ppo_results_sem.append(df_sem)

    plot_modified_ant("ppo", top_hps_seeds, top_ppo_results_mean, top_ppo_results_sem)

    """
    SAC
    """

    # fetchreach_sac_data_dir = DATA_DIR + "/fetchreach/hpsc/sac"
    # fetchreach_sac_data_dirs = os.listdir(fetchreach_sac_data_dir)
    #
    # fetchreach_sac_results = []
    #
    # for dir_ in fetchreach_sac_data_dirs:
    #     fetchreach_sac_result = get_sac_summary_data(fetchreach_sac_data_dir + "/" + dir_)
    #     fetchreach_sac_results.append(fetchreach_sac_result)
    #
    # df = pd.DataFrame(data=fetchreach_sac_results,
    #                   columns=["ps seed",
    #                            "performance mean",
    #                            "ci lower",
    #                            "ci upper",
    #                            "performance_mean_sum"])
    #
    # df = df.sort_values(by=["performance_mean_sum"], ascending=False)
    #
    # df.to_csv(hps_data_dir + "/fetchreach_sac_hps_data.csv", index=False)
    #
    # top_fetchreach_sac_results_mean = []
    # top_fetchreach_sac_results_sem = []
    #
    # df = df["ps seed"].head(NUM_BEST)
    # top_hps_seeds = df.values.tolist()
    #
    # for seed in top_hps_seeds:
    #     for dir_ in fetchreach_sac_data_dirs:
    #         if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
    #             df_mean, df_sem = get_sac_data(fetchreach_sac_data_dir + "/" + dir_)
    #             top_fetchreach_sac_results_mean.append(df_mean)
    #             top_fetchreach_sac_results_sem.append(df_sem)
    #
    # plot_fetchreach("sac", top_hps_seeds, top_fetchreach_sac_results_mean, top_fetchreach_sac_results_sem)


def fetchreach():

    hps_data_dir = os.getcwd() + "/data/fetchreach"
    os.makedirs(hps_data_dir, exist_ok=True)

    """
    PPO
    """

    ppo_data_dir = DATA_DIR + "/fetchreach/hpsc/ppo"
    ppo_data_dirs = os.listdir(ppo_data_dir)

    ppo_results = []

    for dir_ in ppo_data_dirs:
        ppo_result = get_ppo_summary_data(ppo_data_dir + "/" + dir_)
        ppo_results.append(ppo_result)

    df = pd.DataFrame(data=ppo_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/FetchReach-v1_PPO_hps_data.csv", index=False)

    top_ppo_results_mean = []
    top_ppo_results_sem = []

    df = df["ps seed"].head(NUM_BEST)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        for dir_ in ppo_data_dirs:
            if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
                df_mean, df_sem = get_ppo_data(ppo_data_dir + "/" + dir_)
                top_ppo_results_mean.append(df_mean)
                top_ppo_results_sem.append(df_sem)

    plot_fetchreach("ppo", top_hps_seeds, top_ppo_results_mean, top_ppo_results_sem)

    """
    SAC
    """

    fetchreach_sac_data_dir = DATA_DIR + "/fetchreach/hpsc/sac"
    fetchreach_sac_data_dirs = os.listdir(fetchreach_sac_data_dir)

    fetchreach_sac_results = []

    for dir_ in fetchreach_sac_data_dirs:
        fetchreach_sac_result = get_sac_summary_data(fetchreach_sac_data_dir + "/" + dir_)
        fetchreach_sac_results.append(fetchreach_sac_result)

    df = pd.DataFrame(data=fetchreach_sac_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/FetchReach-v1_SAC_hps_data.csv", index=False)

    top_fetchreach_sac_results_mean = []
    top_fetchreach_sac_results_sem = []

    df = df["ps seed"].head(NUM_BEST)
    top_hps_seeds = df.values.tolist()

    for seed in top_hps_seeds:
        for dir_ in fetchreach_sac_data_dirs:
            if "pss:" + str(seed) == dir_[-(4 + len(str(seed))):]:
                df_mean, df_sem = get_sac_data(fetchreach_sac_data_dir + "/" + dir_)
                top_fetchreach_sac_results_mean.append(df_mean)
                top_fetchreach_sac_results_sem.append(df_sem)

    plot_fetchreach("sac", top_hps_seeds, top_fetchreach_sac_results_mean, top_fetchreach_sac_results_sem)


if __name__ == "__main__":

    DATA_DIR = "/mnt/DATA/shared"

    RUNS = 10  # number of runs (seeds) to average across

    NUM_BEST = 2  # plot the "NUM_BEST" best seeds (max: 10)
    assert 1 <= NUM_BEST <= 10, "hps.__main__: NUM_BEST must have a value between 1 and 10"

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:grey","tab:olive", "tab:cyan"]

    ant()
    # fetchreach()