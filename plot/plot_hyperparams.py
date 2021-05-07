import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored

import utils.plot_style_settings as pss

PATH = "/home/sschoepp/Documents/DATA/openai_DATA/hps"

NUM_RUNS = 30

LINE = "--------------------------------------------------------------------------------"

os.makedirs(PATH + "/images", exist_ok=True)


def warning(seed, directory, foldername):

    print(colored("Seed {} is not included in this plot for {}.\nMissing {}.".format(seed, directory, foldername), "red"))


def ppo_plot_data(folder):

    # extract params from the name
    # example name: /home/sschoepp/Documents/DATA/openai_DATA/PPOv2_nao:6000000_arm:right_lr:0.00025_lrd:True_g:0.99_ns:256_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_log_std:0.0_tef:1000_ee:10_hps

    experiment_name = folder[42:]

    parameters = experiment_name.split("_")

    gamma = None
    lr = None
    ns = None
    mbs = None

    for p in parameters:

        if p.startswith("g:"):

            gamma = float(p.split(":")[1])

        elif p.startswith("lr:"):

            lr = float(p.split(":")[1])

        elif p.startswith("ns:"):

            ns = int(p.split(":")[1])

        elif p.startswith("mbs:"):

            mbs = int(p.split(":")[1])

    # obtain all dataframes for all runs of an experiment

    dfs = []
    count = 0  # number of seeds in average

    for i in range(NUM_RUNS):

        seed_foldername = folder + "/seed{}".format(i)

        if os.path.exists(seed_foldername):

            csv_foldername = seed_foldername + "/csv"

            if os.path.exists(csv_foldername):

                csv_filename = csv_foldername + "/eval_data.csv"

                if os.path.exists(csv_filename):

                    count += 1

                    df = pd.read_csv(csv_foldername + "/eval_data.csv")
                    df = df[["num_time_steps", "average_return"]]
                    dfs.append(df)

                else:

                    warning(i, folder, "eval_data.csv file")

            else:

                warning(i, folder, "csv folder")

        else:

            warning(i, folder, "seed folder")

    # average the data

    performance = None
    se_performance = None

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        average_return = df_mean["average_return"]
        se_average_return = df_sem["average_return"]

        performance = average_return[-100:]
        performance = performance.mean()

        se_performance = se_average_return[-100:]
        se_performance = se_performance.sem()

        return gamma, lr, ns, mbs, performance, se_performance, count

    else:

        return


def ppo_plot(data_):

    print("PPO")
    print(LINE)
    print(LINE)

    # data_ is list of tuples (gamma, lr, ns, mbs, performance, se_performance, count)

    # plot gamma and lr=0.00025, ns=128, mbs=32

    gammas = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[0])  # sort by gamma

    for d in data_:

        if d[1] == 0.00025 and d[2] == 256 and d[3] == 64:

            gammas.append(d[0])
            performances.append(d[4])
            se_performances.append(d[5])
            counts.append(d[6])

    gammas = np.array(gammas)
    performances = np.array(performances)
    se_performances = np.array(se_performances)
    counts = np.array(counts)

    print("varying gamma, lr=0.00025, ns=256, mbs=64")
    for i in zip(gammas, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(gammas, performances, marker='o', color="blue")
    plt.fill_between(x=gammas, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("gamma")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("varying gamma, lr=0.00025, ns=256, mbs=64")
    plt.xticks([0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    pss.plot_settings()
    plt.savefig(PATH + "/images/ppo_gamma_lr:0.00025_ns:256_mbs:64.jpg")
    # plt.show()
    plt.close()

    # plot ns=128 and mbs 16, 32, 64, 128

    mbss = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[3])  # sort by mbs

    for d in data_:

        if d[0] == 0.99 and d[2] == 128:
            mbss.append(d[3])
            performances.append(d[4])
            se_performances.append(d[5])
            counts.append(d[6])

    mbss = np.array(mbss)
    performances = np.array(performances)
    se_performances = np.array(se_performances)
    counts = np.array(counts)

    print("gamma: 0.99, ns: 128, varying mbs")
    for i in zip(mbss, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(mbss, performances, marker='o', color="blue")
    plt.fill_between(x=mbss, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("mbs")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("gamma=0.99, ns=128, varying mbs")
    plt.xticks([16, 32, 64, 128])
    pss.plot_settings()
    plt.savefig(PATH + "/images/ppo_gamma:0.99_ns:128_mbs.jpg")
    # plt.show()
    plt.close()

    # plot ns=256 and mbs 32, 64, 128, 256

    mbss = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[3])  # sort by mbs

    for d in data_:

        if d[0] == 0.99 and d[2] == 256:
            mbss.append(d[3])
            performances.append(d[4])
            se_performances.append(d[5])
            counts.append(d[6])

    mbss = np.array(mbss)
    performances = np.array(performances)
    se_performances = np.array(se_performances)
    counts = np.array(counts)

    print("gamma: 0.99, ns: 256, varying mbs")
    for i in zip(mbss, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(mbss, performances, marker='o', color="blue")
    plt.fill_between(x=mbss, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("mbs")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("gamma=0.99, ns=256, varying mbs")
    plt.xticks([32, 64, 128, 256])
    pss.plot_settings()
    plt.savefig(PATH + "/images/ppo_gamma:0.99_ns:256_mbs.jpg")
    # plt.show()
    plt.close()

    # plot ns=512 and mbs 64, 128, 256, 512

    mbss = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[3])  # sort by mbs

    for d in data_:

        if d[0] == 0.99 and d[2] == 512:
            mbss.append(d[3])
            performances.append(d[4])
            se_performances.append(d[5])
            counts.append(d[6])

    mbss = np.array(mbss)
    performances = np.array(performances)
    se_performances = np.array(se_performances)
    counts = np.array(counts)

    print("gamma: 0.99, ns: 512, varying mbs")
    for i in zip(mbss, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(mbss, performances, marker='o', color="blue")
    plt.fill_between(x=mbss, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("mbs")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("gamma=0.99, ns=512, varying mbs")
    plt.xticks([64, 128, 256, 512])
    pss.plot_settings()
    plt.savefig(PATH + "/images/ppo_gamma:0.99_ns:512_mbs.jpg")
    # plt.show()
    plt.close()


def sac_plot_data(folder):

    # extract params from the name
    # example name: /home/sschoepp/Documents/DATA/openai_DATA/SACv2_Ant-v2:12000000_g:0.98_t:0.005_a:0.2_lr:0.0003_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:10000_ee:10_autotune

    experiment_name = folder[42:]

    parameters = experiment_name.split("_")

    gamma = None
    tau = None
    lr = None

    for p in parameters:

        if p.startswith("g:"):

            gamma = float(p.split(":")[1])

        elif p.startswith("t:"):

            tau = float(p.split(":")[1])

        elif p.startswith("lr:"):

            lr = float(p.split(":")[1])

    # obtain all dataframes for all runs of an experiment

    dfs = []
    count = 0  # number of seeds in average

    for i in range(NUM_RUNS):

        seed_foldername = folder + "/seed{}".format(i)

        if os.path.exists(seed_foldername):

            csv_foldername = seed_foldername + "/csv"

            if os.path.exists(csv_foldername):

                csv_filename = csv_foldername + "/eval_data.csv"

                if os.path.exists(csv_filename):

                    count += 1

                    df = pd.read_csv(csv_foldername + "/eval_data.csv")
                    df = df[["num_time_steps", "average_return"]]
                    dfs.append(df)

                else:

                    warning(i, folder, "eval_data.csv file")

            else:

                warning(i, folder, "csv folder")

        else:

            warning(i, folder, "seed folder")

    # average the data

    if len(dfs) > 0:

        df = pd.concat(dfs)
        df = df.groupby(df.index)

        df_mean = df.mean()
        df_sem = df.sem()

        average_return = df_mean["average_return"]
        se_average_return = df_sem["average_return"]

        performance = average_return[-100:]
        performance = performance.mean()

        se_performance = se_average_return[-100:]
        se_performance = se_performance.sem()

        return gamma, tau, lr, performance, se_performance, count

    else:

        return


def sac_plot(data_):

    # data_ is list of tuples (gamma, tau, lr, performance, se_performance, count)

    print(LINE)
    print("SAC")
    print(LINE)
    print(LINE)

    # plot gamma and tau = 0.005

    gammas = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[0])  # sort by gamma

    for d in data_:

        if d[1] == 0.005 and d[2] == 0.0003:

            gammas.append(d[0])
            performances.append(d[3])
            se_performances.append(d[4])
            counts.append(d[5])

    gammas = np.array(gammas)
    performances = np.array(performances)
    se_performances = np.array(se_performances)
    counts = np.array(counts)

    print("varying gamma, tau=0.005")
    for i in zip(gammas, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(gammas, performances, marker='o', color="blue")
    plt.fill_between(x=gammas, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("gamma")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("varying gamma, tau=0.005")
    plt.xticks([0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    pss.plot_settings()
    plt.savefig(PATH + "/images/sac_gamma_tau:0.005.jpg")
    # plt.show()
    plt.close()

    # plot gamma and tau = 0.01

    gammas = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[0])  # sort by gamma

    for d in data_:

        if d[1] == 0.01 and d[2] == 0.0003:
            gammas.append(d[0])
            performances.append(d[3])
            se_performances.append(d[4])
            counts.append(d[5])

    gammas = np.array(gammas)
    performances = np.array(performances)
    se_performances = np.array(se_performances)

    print("varying gamma, tau=0.01")
    for i in zip(gammas, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(gammas, performances, marker='o', color="blue")
    plt.fill_between(x=gammas, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("gamma")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("varying gamma, tau=0.01")
    plt.xticks([0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    pss.plot_settings()
    plt.savefig(PATH + "/images/sac_gamma_tau:0.01.jpg")
    # plt.show()
    plt.close()

    # plot gamma and tau = 0.015

    gammas = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[0])  # sort by gamma

    for d in data_:

        if d[1] == 0.015 and d[2] == 0.0003:
            gammas.append(d[0])
            performances.append(d[3])
            se_performances.append(d[4])
            counts.append(d[5])

    gammas = np.array(gammas)
    performances = np.array(performances)
    se_performances = np.array(se_performances)

    print("varying gamma, tau=0.015")
    for i in zip(gammas, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(gammas, performances, marker='o', color="blue")
    plt.fill_between(x=gammas, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("gamma")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("varying gamma, tau=0.015")
    plt.xticks([0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    pss.plot_settings()
    plt.savefig(PATH + "/images/sac_gamma_tau:0.015.jpg")
    # plt.show()
    plt.close()

    # plot lr and gamma=0.94, tau=0.01

    lrs = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[2])  # sort by lr

    for d in data_:

        if d[0] == 0.94 and d[1] == 0.01:
            lrs.append(d[2])
            performances.append(d[3])
            se_performances.append(d[4])
            counts.append(d[5])

    lrs = np.array(lrs)
    performances = np.array(performances)
    se_performances = np.array(se_performances)

    print("varying lr, gamma=0.94, tau=0.01")
    for i in zip(lrs, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(lrs, performances, marker='o', color="blue")
    plt.fill_between(x=lrs, y1=(performances - se_performances), y2=(performances + se_performances), color="blue", alpha=0.2)
    plt.xlabel("learning rate")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("varying lr, gamma=0.94, tau=0.01")
    plt.xticks([0.0003, 0.00045, 0.0006])
    pss.plot_settings()
    plt.savefig(PATH + "/images/sac_lr_gamma:0.94_tau:0.01.jpg")
    # plt.show()
    plt.close()

    # plot lr and gamma=0.96, tau=0.01

    lrs = []
    performances = []
    se_performances = []
    counts = []

    data_.sort(key=lambda tup: tup[2])  # sort by lr

    for d in data_:

        if d[0] == 0.96 and d[1] == 0.01:
            lrs.append(d[2])
            performances.append(d[3])
            se_performances.append(d[4])
            counts.append(d[5])

    lrs = np.array(lrs)
    performances = np.array(performances)
    se_performances = np.array(se_performances)

    print("varying lr, gamma=0.96, tau=0.01")
    for i in zip(lrs, performances, se_performances, counts):
        print(i)
    print(LINE)

    plt.plot(lrs, performances, marker='o', color="blue")
    plt.fill_between(x=lrs, y1=(performances - se_performances), y2=(performances + se_performances), color="blue",
                     alpha=0.2)
    plt.xlabel("learning rate")
    plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
    plt.title("varying lr, gamma=0.96, tau=0.01")
    plt.xticks([0.0003, 0.00045, 0.0006])
    pss.plot_settings()
    plt.savefig(PATH + "/images/sac_lr_gamma:0.96_tau:0.01.jpg")
    # plt.show()
    plt.close()


if __name__ == "__main__":

    sac_data = []

    ppo_data = []

    dirs = os.listdir(PATH)

    for dir_ in dirs:

        prefix = dir_[0:3]

        if prefix == "SAC":

            data_point = sac_plot_data(PATH + "/" + dir_)

            sac_data.append(data_point)

        elif prefix == "PPO":

            data_point = ppo_plot_data(PATH + "/" + dir_)

            ppo_data.append(data_point)

    # we have all our data points to plot

    sac_plot(sac_data)
    ppo_plot(ppo_data)
