import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def get_data(directory):
    """
    Obtain the average return across 10 runs for a single experiment.

    @param directory: string
        directory for experiment data
    """

    dfs = []

    num_runs = 5

    for i in range(num_runs):

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
        df_sem = df.sem()

        return df_mean, df_sem


def plot(directory, algorithm, df_mean, df_sem):
    """
    Plot experiment.

    @param algorithm: string
        algorithm name
    @param df_mean: df
        average return for one parameter setting
    @param df_sem: df
        standard error for one parameter setting
    """

    plot_directory = os.getcwd() + "/plots"
    os.makedirs(plot_directory, exist_ok=True)

    ymin = -2000  # min for y axis
    ymax = 8000  # max for y axis

    title = "test"

    x = df_mean.index

    y = df_mean.reset_index()["average_return"]
    lb = y - CI_Z * df_sem.reset_index()["average_return"]
    ub = y + CI_Z * df_sem.reset_index()["average_return"]

    plt.plot(x, y, color="tab:blue")
    plt.fill_between(x, lb, ub, color="tab:blue", alpha=0.3)

    plt.xlabel("time steps")
    plt.ylim(ymin, ymax)
    plt.ylabel("average\nreturn\n(5 seeds)", labelpad=35).set_rotation(0)
    plt.title("Ant-v2 {}".format(algorithm.upper()), fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_directory + "/{}_{}.jpg".format(title, algorithm.upper()))
    plt.show()
    plt.close()


if __name__ == "__main__":

    CI_Z = 2.262

    DATA_DIR = "/mnt/DATA/lr/PPOv2_Ant-v2:400000000_lr:0.000123_lrd:True_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_r_0.5"

    df_mean, df_sem = get_data(DATA_DIR)

    plot(DATA_DIR, "ppo", df_mean, df_sem)
