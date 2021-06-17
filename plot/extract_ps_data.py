import csv
import numpy as np
import os
import pandas as pd


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

        seed_foldername = DATA_DIR + "/sac/" + directory + "/seed{}".format(i)
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

        # confidence interval calculation: 9 degrees of freedom, 95% confidence (or alpha=0.025), table value is 2.262
        # https://www.statisticshowto.com/probability-and-statistics/confidence-interval/
        sac_results.append((pss, g, lr, t, rbs, bs, performance_mean, performance_std, performance_sem, performance_mean - 2.262*performance_sem, performance_mean + 2.262*performance_sem))


if __name__ == "__main__":

    DATA_DIR = "/local/melco2-1/shared/ant"
    RUNS = 10

    sac_dirs = os.listdir(DATA_DIR + "/sac")

    sac_results = []

    for dir_ in sac_dirs:

        extract_sac_results(dir_)

    # sort by descending performance mean
    def sort_descending(sac_result):
        return sac_result[6]
    sac_results.sort(key=sort_descending)

    df = pd.DataFrame(data=sac_results,
                      columns=["ps seed", "gamma", "learning rate", "tau", "replay buffer size", "batch size", "performance mean", "performance std", "performance sem", "ci lower", "ci upper"])

    df.to_csv("sac_param_search.csv", index=False)