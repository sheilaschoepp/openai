import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

sns.set_theme()


def plot(directory):
    """
    Obtain the average return across multiple runs for an experiment.
    Plot experiment.

    @param directory: string
        absolute path for experiment seed directory
    """

    num_seeds = 10

    dfs = []

    for s in os.listdir(directory):

        seed_foldername = directory + "/{}".format(s)
        csv_filename = seed_foldername + "/csv/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)
            df = df[["num_time_steps", "average_return"]].set_index("num_time_steps")
            dfs.append(df)

    if len(dfs) < num_seeds:

        print(colored("The number of seeds for this experiemnt is less than 10", "red"))

    df = pd.concat(dfs)
    df = df.groupby(df.index)

    df_mean = df.mean()
    df_sem = df.sem()

    num_seeds = len(dfs)



def plot(title, df_mean, df_sem, num_seeds, ymin, ymax):
    """
    Plot experiment.

    @param title: string
        title for the plot
    @param df_mean: df
        average return for one parameter setting
    @param df_sem: df
        standard error for one parameter setting
    @param num_seeds: int
        number of seeds used in average
    @param ymin: int
        minimum y value on y-axis
    @param ymax: int
        maximum y value on y-axis
    """

    plot_directory = os.getcwd() + "/plots"
    os.makedirs(plot_directory, exist_ok=True)

    x = df_mean.index

    y = df_mean.reset_index()["average_return"]
    lb = y - CI_Z * df_sem.reset_index()["average_return"]
    ub = y + CI_Z * df_sem.reset_index()["average_return"]

    plt.plot(x, y, color="tab:blue")
    plt.fill_between(x, lb, ub, color="tab:blue", alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n({} seeds)".format(num_seeds), labelpad=35).set_rotation(0)
    plt.title("{}".format(title), fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_directory + "/{}.jpg".format(title))
    # plt.show()
    plt.close()


if __name__ == "__main__":

    CI_Z = 2.262