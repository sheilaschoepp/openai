import csv
import numpy as np
import os
import pandas as pd


def extract_ppo_results(directory):
    """

    @param directory: string
        directory for PPO experiment data

    Example: PPOv2_Ant-v2:1000000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu
    """

    parameters = directory.split("_")

    ns = None
    mbs = None
    epo = None
    eps = None
    g = None
    lam = None
    c1 = None
    c2 = None
    lr = None
    pss = None

    for p in parameters:

        if p.startswith("ns:"):
            ns = int(p.split(":")[1])

        elif p.startswith("mbs:"):
            mbs = int(p.split(":")[1])

        elif p.startswith("epo:"):
            epo = int(p.split(":")[1])

        elif p.startswith("eps:"):
            eps = float(p.split(":")[1])

        elif p.startswith("g:"):
            g = float(p.split(":")[1])

        elif p.startswith("lam:"):
            lam = float(p.split(":")[1])

        elif p.startswith("c1:"):
            c1 = float(p.split(":")[1])

        elif p.startswith("c2:"):
            c2 = float(p.split(":")[1])

        elif p.startswith("lr:"):
            lr = float(p.split(":")[1])

        elif p.startswith("pss:"):
            pss = int(p.split(":")[1])

    dfs = []

    for i in range(RUNS):

        seed_foldername = directory + "/seed{}".format(i)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df["average_return"]
            dfs.append(df)

        else:

            print("missing:", seed_foldername)

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


def extract_sac_results(directory):
    """

    @param directory: string
        directory for SAC experiment data

    Example: SACv2_Ant-v2:5000000_g:0.91_t:0.0292_a:0.2_lr:0.00425_hd:256_rbs:10000_bs:64_mups:1_tui:1_tef:10000_ee:10_tmsf:100000_a:True_d:cuda_ps:True_pss:3_r
    """

    parameters = directory.split("_")

    g = None
    lr = None
    t = None
    rbs = None
    bs = None
    pss = None

    for p in parameters:

        if p.startswith("g:"):
            g = float(p.split(":")[1])

        elif p.startswith("lr:"):
            lr = float(p.split(":")[1])

        elif p.startswith("t:"):
            t = float(p.split(":")[1])

        elif p.startswith("rbs:"):
            rbs = int(p.split(":")[1])

        elif p.startswith("bs:"):
            bs = int(p.split(":")[1])

        elif p.startswith("pss:"):
            pss = int(p.split(":")[1])

    dfs = []

    for i in range(RUNS):

        seed_foldername = directory + "/seed{}".format(i)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df["average_return"]
            dfs.append(df)

        else:
            
            print("missing:", seed_foldername)

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


def ant():

    hps_data_dir = os.getcwd() + "/numerical_hps_results/ant"
    os.makedirs(hps_data_dir, exist_ok=True)

    ant_data_dir = DATA_DIR + "/ant"

    ant_data_dirs = os.listdir(ant_data_dir)

    """
    PPO
    """

    ant_ppo_results = []

    for dir_ in ant_data_dirs:
        if "PPO" in dir_:
            ant_ppo_result = extract_ppo_results(ant_data_dir + "/" + dir_)
            ant_ppo_results.append(ant_ppo_result)

    df = pd.DataFrame(data=ant_ppo_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/ant_ppo_hps_data.csv", index=False)



    """
    SAC
    """

    ant_sac_results = []

    for dir_ in ant_data_dirs:
        if "SAC" in dir_:
            ant_sac_result = extract_sac_results(ant_data_dir + "/" + dir_)
            ant_sac_results.append(ant_sac_result)

    df = pd.DataFrame(data=ant_sac_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/ant_sac_hps_data.csv", index=False)


def fetchreach():

    hps_data_dir = os.getcwd() + "/numerical_hps_results/fetchreach"
    os.makedirs(hps_data_dir, exist_ok=True)

    fetchreach_data_dir = DATA_DIR + "/fetchreach"

    fetchreach_data_dirs = os.listdir(fetchreach_data_dir)

    """
    PPO
    """

    fetchreach_ppo_results = []

    for dir_ in fetchreach_data_dirs:
        if "PPO" in dir_:
            fetchreach_ppo_result = extract_ppo_results(fetchreach_data_dir + "/" + dir_)
            fetchreach_ppo_results.append(fetchreach_ppo_result)

    df = pd.DataFrame(data=fetchreach_ppo_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/fetchreach_ppo_hps_data.csv", index=False)

    """
    SAC
    """

    fetchreach_sac_results = []

    for dir_ in fetchreach_data_dirs:
        if "SAC" in dir_:
            fetchreach_sac_result = extract_sac_results(fetchreach_data_dir + "/" + dir_)
            fetchreach_sac_results.append(fetchreach_sac_result)

    df = pd.DataFrame(data=fetchreach_sac_results,
                      columns=["ps seed",
                               "performance mean",
                               "ci lower",
                               "ci upper",
                               "performance_mean_sum"])

    df = df.sort_values(by=["performance_mean_sum"], ascending=False)

    df.to_csv(hps_data_dir + "/fetchreach_sac_hps_data.csv", index=False)


if __name__ == "__main__":

    DATA_DIR = "/Users/sheilaschoepp/Documents/DATA"

    RUNS = 10

    ant()
    fetchreach()