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

    for i in os.listdir(directory):

        seed_foldername = directory + "/{}".format(i)
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


def plot(title, df_mean, df_sem, ymin, ymax):
    """
    Plot experiment.

    @param title: string
        title for the plot
    @param algorithm: string
        algorithm name
    @param df_mean: df
        average return for one parameter setting
    @param df_sem: df
        standard error for one parameter setting
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
    plt.ylabel("average\nreturn\n(5 seeds)", labelpad=35).set_rotation(0)
    plt.title("{}".format(title), fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_directory + "/{}.jpg".format(title))
    plt.show()
    plt.close()


if __name__ == "__main__":

    CI_Z = 2.262

    data_dir = "/mnt/DATA/shared/ant/lr/PPOv2_Ant-v2:400000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu_r_0.5"

    df_mean, df_sem = get_data(data_dir)

    plot("ppo: (1) 50% lr reduction (2) default param ", df_mean, df_sem, -1000, 8000)

    data_dir = "/mnt/DATA/shared/ant/lr/PPOv2_Ant-v2:400000000_lr:0.000123_lrd:True_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:5000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_r_0.5"

    df_mean, df_sem = get_data(data_dir)

    plot("ppo: (1) 50% lr reduction (2) pss 33 ", df_mean, df_sem, -1000, 8000)

    data_dir = "/mnt/DATA/shared/ant/faulty/test/SACv2_AntEnv-v1:2000000_Ant-v2:25000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_r"

    df_mean, df_sem = get_data(data_dir)

    plot("sac: AntEnv-v1", df_mean, df_sem, -1000, 8000)

    data_dir = "/mnt/DATA/shared/ant/faulty/test/SACv2_AntEnv-v2:2000000_Ant-v2:25000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_r"

    df_mean, df_sem = get_data(data_dir)

    plot("sac: AntEnv-v2", df_mean, df_sem, -1000, 8000)

    data_dir = "/mnt/DATA/shared/ant/faulty/test/SACv2_AntEnv-v3:2000000_Ant-v2:25000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_r"

    df_mean, df_sem = get_data(data_dir)

    plot("sac: AntEnv-v3", df_mean, df_sem, -1000, 8000)

    data_dir = "/mnt/DATA/shared/ant/faulty/test/SACv2_AntEnv-v4:2000000_Ant-v2:25000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_r"

    df_mean, df_sem = get_data(data_dir)

    plot("sac: AntEnv-v4", df_mean, df_sem, -1000, 8000)

    data_dir = "/mnt/DATA/shared/fetchreach/goal_dist/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43_0.001"

    df_mean, df_sem = get_data(data_dir)

    plot("ppo: goal dist 0.001", df_mean, df_sem, -15, 5)

    data_dir = "/mnt/DATA/shared/fetchreach/goal_dist/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:43_0.005"

    df_mean, df_sem = get_data(data_dir)

    plot("ppo: goal dist 0.005", df_mean, df_sem, -15, 5)

    data_dir = "/mnt/DATA/shared/fetchreach/goal_dist/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:21_0.001"

    df_mean, df_sem = get_data(data_dir)

    plot("sac: goal dist 0.001", df_mean, df_sem, -15, 5)

    data_dir = "/mnt/DATA/shared/fetchreach/goal_dist/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:21_0.005"

    df_mean, df_sem = get_data(data_dir)

    plot("sac: goal dist 0.005", df_mean, df_sem, -15, 5)
