import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored

import utils.plot_style_settings as pss

"""
Plotting.
"""

DATA_DIR = "/media/sschoepp/easystore/melco/complete/"

RUNS = 30

STEP = 1

file = open("experiment_data.txt", "w")

"""
Combined SAC plots.
"""

"""
AntEnv-v1
"""

# CRB=F

EXPERIMENT = "SACv2_AntEnv-v1:25000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# CRB=T

EXPERIMENT = "SACv2_AntEnv-v1:25000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:True_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:True_autotune_d:cuda_resumed"

dfs2 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs2.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df2 = pd.concat(dfs2)
df2 = df2.groupby(df2.index)

df2_median = df2.median().iloc[::STEP, :]
df2_std = df2.std().iloc[::STEP, :]

x2 = df2_median.iloc[:, 1].to_numpy()
median2 = df2_median.iloc[:, -1].to_numpy()
std2 = df2_std.iloc[:, -1].to_numpy()
sem2 = std2 / np.sqrt(30)

lower2 = median2 - 2.045 * sem2
upper2 = median2 + 2.045 * sem2

# network reinitialization

EXPERIMENT = "SACv2_AntEnv-v1:25000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v1:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"

dfs3 = []

post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 4].mean()
            partial_avg = df.iloc[150:160, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs3.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df3 = pd.concat(dfs3)
df3 = df3.groupby(df3.index)

df3_median = df3.median().iloc[::STEP, :]
df3_std = df3.std().iloc[::STEP, :]

x3 = df3_median.iloc[:, 1].to_numpy()
median3 = df3_median.iloc[:, -1].to_numpy()
std3 = df3_std.iloc[:, -1].to_numpy()
sem3 = std3 / np.sqrt(30)

lower3 = median3 - 2.045 * sem3
upper3 = median3 + 2.045 * sem3

"""
Plots of all three.
"""

fig = plt.figure()
ax = plt.plot(x1, median1, color="saddlebrown", label="rb & models")
plt.fill_between(x1, lower1, upper1, color="saddlebrown", alpha=0.2)
ax2 = plt.plot(x2, median2, color="blue", label="models")
plt.fill_between(x2, lower2, upper2, color="blue", alpha=0.2)
ax3 = plt.plot(x3+20000000, median3, color="green", label="nothing")
plt.fill_between(x3+20000000, lower3, upper3, color="green", alpha=0.2)
plt.axvline(x=20000000, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
# plt.title("SAC in Ant-v2: Broken, Severed Effector")
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v1_sac_all3.jpg".format(EXPERIMENT))
plt.show()
plt.close()

"""
AntEnv-v2
"""

# CRB=F

EXPERIMENT = "SACv2_AntEnv-v2:25000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# CRB=T  # TODO

# EXPERIMENT = ""
# EXPERIMENT = "SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:True_autotune_d:cuda_resumed"
#
# dfs2 = []
#
# pre = []  # prior to fault
# post = []  # after a fault
# partial = []  # after partial adaptation (300k time steps)
# final = []  # after learning is complete
#
# for seed in range(RUNS):
#
#     seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
#     csv_foldername = seed_data_dir + "/csv"
#
#     if os.path.exists(csv_foldername):
#
#         csv_filename = csv_foldername + "/eval_data.csv"
#
#         if os.path.exists(csv_filename):
#
#             df = pd.read_csv(csv_filename)
#
#             pre_avg = df.iloc[391:401, 4].mean()
#             post_avg = df.iloc[401:411, 4].mean()
#             partial_avg = df.iloc[801:811, 4].mean()
#             final_avg = df.iloc[-10:, 4].mean()
#
#             pre.append(pre_avg)
#             post.append(post_avg)
#             partial.append(partial_avg)
#             final.append(final_avg)
#
#             dfs2.append(df)
#
#         else:
#
#             print(colored(csv_filename + " file is missing from the plot", "red"))
#
#     else:
#
#         print(colored(csv_foldername + " folder is missing from the plot", "red"))
#
# pre = pd.DataFrame(pre)
# post = pd.DataFrame(post)
# partial = pd.DataFrame(partial)
# final = pd.DataFrame(final)
#
# post_diff = (post - pre)
# post_diff_mean = round(post_diff.mean()[0], 4)
# post_diff_sem = round(post_diff.sem()[0], 4)
# post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
# post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
# post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)
#
# partial_diff = (partial - pre)
# partial_diff_mean = round(partial_diff.mean()[0], 4)
# partial_diff_sem = round(partial_diff.sem()[0], 4)
# partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
# partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
# partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)
#
# final_diff = (final - pre)
# final_diff_mean = round(final_diff.mean()[0], 4)
# final_diff_sem = round(final_diff.sem()[0], 4)
# final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
# final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
# final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)
#
# file.write(EXPERIMENT + "\n")
# file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
# file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
# file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))
#
# df2 = pd.concat(dfs2)
# df2 = df2.groupby(df2.index)
#
# df2_median = df2.median().iloc[::STEP, :]
# df2_std = df2.std().iloc[::STEP, :]
#
# x2 = df2_median.iloc[:, 1].to_numpy()
# median2 = df2_median.iloc[:, -1].to_numpy()
# std2 = df2_std.iloc[:, -1].to_numpy()
# sem2 = std2 / np.sqrt(30)
#
# lower2 = median2 - 2.045 * sem2
# upper2 = median2 + 2.045 * sem2

# network reinitialization

EXPERIMENT = "SACv2_AntEnv-v2:25000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"

dfs3 = []

post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 4].mean()
            partial_avg = df.iloc[150:160, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs3.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df3 = pd.concat(dfs3)
df3 = df3.groupby(df3.index)

df3_median = df3.median().iloc[::STEP, :]
df3_std = df3.std().iloc[::STEP, :]

x3 = df3_median.iloc[:, 1].to_numpy()
median3 = df3_median.iloc[:, -1].to_numpy()
std3 = df3_std.iloc[:, -1].to_numpy()
sem3 = std3 / np.sqrt(30)

lower3 = median3 - 2.045 * sem3
upper3 = median3 + 2.045 * sem3

"""
Plots of all three.
"""

fig = plt.figure()
ax = plt.plot(x1, median1, color="saddlebrown", label="rb & models")
plt.fill_between(x1, lower1, upper1, color="saddlebrown", alpha=0.2)
ax2 = plt.plot(x2, median2, color="blue", label="models")
plt.fill_between(x2, lower2, upper2, color="blue", alpha=0.2)
ax3 = plt.plot(x3+20000000, median3, color="green", label="nothing")
plt.fill_between(x3+20000000, lower3, upper3, color="green", alpha=0.2)
plt.axvline(x=20000000, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
# plt.title("SAC in Ant-v2: Hip ROM Decrease")
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v2_sac_all3.jpg".format(EXPERIMENT))
plt.show()
plt.close()

"""
AntEnv-v3
"""

# CRB=F

EXPERIMENT = "SACv2_AntEnv-v3:25000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# CRB=T

EXPERIMENT = "SACv2_AntEnv-v3:25000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:True_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:True_autotune_d:cuda_resumed"

dfs2 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs2.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df2 = pd.concat(dfs2)
df2 = df2.groupby(df2.index)

df2_median = df2.median().iloc[::STEP, :]
df2_std = df2.std().iloc[::STEP, :]

x2 = df2_median.iloc[:, 1].to_numpy()
median2 = df2_median.iloc[:, -1].to_numpy()
std2 = df2_std.iloc[:, -1].to_numpy()
sem2 = std2 / np.sqrt(30)

lower2 = median2 - 2.045 * sem2
upper2 = median2 + 2.045 * sem2

# network reinitialization

EXPERIMENT = "SACv2_AntEnv-v3:25000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"
# EXPERIMENT = "SACv2_AntEnv-v3:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"

dfs3 = []

post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 4].mean()
            partial_avg = df.iloc[150:160, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs3.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df3 = pd.concat(dfs3)
df3 = df3.groupby(df3.index)

df3_median = df3.median().iloc[::STEP, :]
df3_std = df3.std().iloc[::STEP, :]

x3 = df3_median.iloc[:, 1].to_numpy()
median3 = df3_median.iloc[:, -1].to_numpy()
std3 = df3_std.iloc[:, -1].to_numpy()
sem3 = std3 / np.sqrt(30)

lower3 = median3 - 2.045 * sem3
upper3 = median3 + 2.045 * sem3

"""
Plots of all three.
"""

fig = plt.figure()
ax = plt.plot(x1, median1, color="saddlebrown", label="rb & models")
plt.fill_between(x1, lower1, upper1, color="saddlebrown", alpha=0.2)
ax2 = plt.plot(x2, median2, color="blue", label="models")
plt.fill_between(x2, lower2, upper2, color="blue", alpha=0.2)
ax3 = plt.plot(x3+20000000, median3, color="green", label="nothing")
plt.fill_between(x3+20000000, lower3, upper3, color="green", alpha=0.2)
plt.axvline(x=20000000, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
# plt.title("SAC in Ant-v2: Ankle ROM Decrease")
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v3_sac_all3.jpg".format(EXPERIMENT))
plt.show()
plt.close()

"""
AntEnv-v5
"""

# CRB=F

EXPERIMENT = "SACv2_AntEnv-v5:20000000_AntEnv-v4:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:False_autotune_d:cuda_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# CRB=T

EXPERIMENT = "SACv2_AntEnv-v5:20000000_AntEnv-v4:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_crb:True_autotune_d:cuda_resumed"

dfs2 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[391:401, 4].mean()
            post_avg = df.iloc[401:411, 4].mean()
            partial_avg = df.iloc[801:811, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs2.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df2 = pd.concat(dfs2)
df2 = df2.groupby(df2.index)

df2_median = df2.median().iloc[::STEP, :]
df2_std = df2.std().iloc[::STEP, :]

x2 = df2_median.iloc[:, 1].to_numpy()
median2 = df2_median.iloc[:, -1].to_numpy()
std2 = df2_std.iloc[:, -1].to_numpy()
sem2 = std2 / np.sqrt(30)

lower2 = median2 - 2.045 * sem2
upper2 = median2 + 2.045 * sem2

# network reinitialization

EXPERIMENT = "SACv2_AntEnv-v5:20000000_g:0.96_t:0.01_a:0.2_lr:0.00045_hd:256_rbs:1000000_bs:256_mups:1_tui:1_tef:50000_ee:10_tmsf:1000000_autotune_d:cuda_resumed"

dfs3 = []

post = []  # after a fault
partial = []  # after partial adaptation (300k time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 4].mean()
            partial_avg = df.iloc[150:160, 4].mean()
            final_avg = df.iloc[-10:, 4].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs3.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df3 = pd.concat(dfs3)
df3 = df3.groupby(df3.index)

df3_median = df3.median().iloc[::STEP, :]
df3_std = df3.std().iloc[::STEP, :]

x3 = df3_median.iloc[:, 1].to_numpy()
median3 = df3_median.iloc[:, -1].to_numpy()
std3 = df3_std.iloc[:, -1].to_numpy()
sem3 = std3 / np.sqrt(30)

lower3 = median3 - 2.045 * sem3
upper3 = median3 + 2.045 * sem3

"""
Plots of all three.
"""

fig = plt.figure()
ax = plt.plot(x1, median1, color="saddlebrown", label="rb & models")
plt.fill_between(x1, lower1, upper1, color="saddlebrown", alpha=0.2)
ax2 = plt.plot(x2, median2, color="blue", label="models")
plt.fill_between(x2, lower2, upper2, color="blue", alpha=0.2)
ax3 = plt.plot(x3+20000000, median3, color="green", label="nothing")
plt.fill_between(x3+20000000, lower3, upper3, color="green", alpha=0.2)
plt.axvline(x=20000000, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v5_sac_all3.jpg".format(EXPERIMENT))
# plt.show()
plt.close()

file.write("\n")

"""
Combined PPO plots.
"""

steps = 400000000

"""
AntEnv-v1
"""

EXPERIMENT = "PPOv2_AntEnv-v1:400000000_Ant-v2:400000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"
# EXPERIMENT = "PPOv2_AntEnv-v1:200000000_Ant-v2:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (2 million time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[991:1001, 6].mean()
            post_avg = df.iloc[1001:1011, 6].mean()
            partial_avg = df.iloc[801:811, 6].mean()
            final_avg = df.iloc[-10:, 6].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# PPO network reinitialization

EXPERIMENT = "PPOv2_AntEnv-v1:400000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"
# EXPERIMENT = "PPOv2_AntEnv-v1:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"

dfs2 = []

post = []  # after a fault
partial = []  # after partial adaptation (2 million time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 6].mean()
            partial_avg = df.iloc[200:210, 6].mean()
            final_avg = df.iloc[-10:, 6].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs2.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df2 = pd.concat(dfs2)
df2 = df2.groupby(df2.index)

df2_median = df2.median().iloc[::STEP, :]
df2_std = df2.std().iloc[::STEP, :]

x2 = df2_median.iloc[:, 1].to_numpy()
median2 = df2_median.iloc[:, -1].to_numpy()
std2 = df2_std.iloc[:, -1].to_numpy()
sem2 = std2 / np.sqrt(20)

lower2 = median2 - 2.045 * sem2
upper2 = median2 + 2.045 * sem2

fig = plt.figure()
ax = plt.plot(x1, median1, color="blue", label="models")
plt.fill_between(x1, lower1, upper1, color="blue", alpha=0.2)
ax2 = plt.plot(x2+steps, median2, color="green", label="nothing")
plt.fill_between(x2+steps, lower2, upper2, color="green", alpha=0.2)
plt.axvline(x=steps, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
# plt.title("PPO in Ant-v2: Broken, Severed Effector")
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v1_ppo_all2.jpg".format(EXPERIMENT))
plt.show()
plt.close()

"""
AntEnv-v2
"""

EXPERIMENT = "PPOv2_AntEnv-v2:400000000_Ant-v2:400000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"
# EXPERIMENT = "PPOv2_AntEnv-v2:200000000_Ant-v2:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (2 million time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[991:1001, 6].mean()
            post_avg = df.iloc[1001:1011, 6].mean()
            partial_avg = df.iloc[801:811, 6].mean()
            final_avg = df.iloc[-10:, 6].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# PPO network reinitialization

EXPERIMENT = "PPOv2_AntEnv-v2:400000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"
# EXPERIMENT = "PPOv2_AntEnv-v2:200000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"

dfs2 = []

post = []  # after a fault
partial = []  # after partial adaptation (2 million time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 6].mean()
            partial_avg = df.iloc[200:210, 6].mean()
            final_avg = df.iloc[-10:, 6].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs2.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df2 = pd.concat(dfs2)
df2 = df2.groupby(df2.index)

df2_median = df2.median().iloc[::STEP, :]
df2_std = df2.std().iloc[::STEP, :]

x2 = df2_median.iloc[:, 1].to_numpy()
median2 = df2_median.iloc[:, -1].to_numpy()
std2 = df2_std.iloc[:, -1].to_numpy()
sem2 = std2 / np.sqrt(20)

lower2 = median2 - 2.045 * sem2
upper2 = median2 + 2.045 * sem2

fig = plt.figure()
ax = plt.plot(x1, median1, color="blue", label="models")
plt.fill_between(x1, lower1, upper1, color="blue", alpha=0.2)
ax2 = plt.plot(x2+steps, median2, color="green", label="nothing")
plt.fill_between(x2+steps, lower2, upper2, color="green", alpha=0.2)
plt.axvline(x=steps, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
# plt.title("PPO in Ant-v2: Hip ROM Decrease")
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v2_ppo_all2.jpg".format(EXPERIMENT))
plt.show()
plt.close()

"""
AntEnv-v3
"""

EXPERIMENT = "PPOv2_AntEnv-v3:400000000_Ant-v2:400000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"
# EXPERIMENT = "PPOv2_AntEnv-v3:200000000_Ant-v2:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"

dfs1 = []

pre = []  # prior to fault
post = []  # after a fault
partial = []  # after partial adaptation (2 million time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            pre_avg = df.iloc[991:1001, 6].mean()
            post_avg = df.iloc[1001:1011, 6].mean()
            partial_avg = df.iloc[801:811, 6].mean()
            final_avg = df.iloc[-10:, 6].mean()

            pre.append(pre_avg)
            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs1.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

pre = pd.DataFrame(pre)
post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df1 = pd.concat(dfs1)
df1 = df1.groupby(df1.index)

df1_median = df1.median().iloc[::STEP, :]
df1_std = df1.std().iloc[::STEP, :]

x1 = df1_median.iloc[:, 1].to_numpy()
median1 = df1_median.iloc[:, -1].to_numpy()
std1 = df1_std.iloc[:, -1].to_numpy()
sem1 = std1 / np.sqrt(30)

lower1 = median1 - 2.045 * sem1
upper1 = median1 + 2.045 * sem1

# PPO network reinitialization

EXPERIMENT = "PPOv2_AntEnv-v3:400000000_lr:0.00025_lrd:False_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"
# EXPERIMENT = "PPOv2_AntEnv-v3:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"

dfs2 = []

post = []  # after a fault
partial = []  # after partial adaptation (2 million time steps)
final = []  # after learning is complete

for seed in range(RUNS):

    seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
    csv_foldername = seed_data_dir + "/csv"

    if os.path.exists(csv_foldername):

        csv_filename = csv_foldername + "/eval_data.csv"

        if os.path.exists(csv_filename):

            df = pd.read_csv(csv_filename)

            post_avg = df.iloc[:10, 6].mean()
            partial_avg = df.iloc[200:210, 6].mean()
            final_avg = df.iloc[-10:, 6].mean()

            post.append(post_avg)
            partial.append(partial_avg)
            final.append(final_avg)

            dfs2.append(df)

        else:

            print(colored(csv_filename + " file is missing from the plot", "red"))

    else:

        print(colored(csv_foldername + " folder is missing from the plot", "red"))

post = pd.DataFrame(post)
partial = pd.DataFrame(partial)
final = pd.DataFrame(final)

post_diff = (post - pre)
post_diff_mean = round(post_diff.mean()[0], 4)
post_diff_sem = round(post_diff.sem()[0], 4)
post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)

partial_diff = (partial - pre)
partial_diff_mean = round(partial_diff.mean()[0], 4)
partial_diff_sem = round(partial_diff.sem()[0], 4)
partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)

final_diff = (final - pre)
final_diff_mean = round(final_diff.mean()[0], 4)
final_diff_sem = round(final_diff.sem()[0], 4)
final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)

file.write(EXPERIMENT + "\n")
file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))

df2 = pd.concat(dfs2)
df2 = df2.groupby(df2.index)

df2_median = df2.median().iloc[::STEP, :]
df2_std = df2.std().iloc[::STEP, :]

x2 = df2_median.iloc[:, 1].to_numpy()
median2 = df2_median.iloc[:, -1].to_numpy()
std2 = df2_std.iloc[:, -1].to_numpy()
sem2 = std2 / np.sqrt(20)

lower2 = median2 - 2.045 * sem2
upper2 = median2 + 2.045 * sem2

fig = plt.figure()
ax = plt.plot(x1, median1, color="blue", label="models")
plt.fill_between(x1, lower1, upper1, color="blue", alpha=0.2)
ax2 = plt.plot(x2+steps, median2, color="green", label="nothing")
plt.fill_between(x2+steps, lower2, upper2, color="green", alpha=0.2)
plt.axvline(x=steps, alpha=1.0, color="red", ymax=0.05)
plt.legend(loc="lower right")
plt.ylim(-1000, 9000)
pss.plot_settings()
# plt.title("PPO in Ant-v2: Ankle ROM Decrease")
plt.xlabel("time steps")
plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)

plt.savefig(DATA_DIR + "images/v3_ppo_all2.jpg".format(EXPERIMENT))
plt.show()
plt.close()

# """
# AntEnv-v5
# """
#
# EXPERIMENT = "PPOv2_AntEnv-v5:200000000_AntEnv-v4:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_cm:False_d:cpu_resumed"
#
# dfs1 = []
#
# pre = []  # prior to fault
# post = []  # after a fault
# partial = []  # after partial adaptation (2 million time steps)
# final = []  # after learning is complete
#
# for seed in range(RUNS):
#
#     seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
#     csv_foldername = seed_data_dir + "/csv"
#
#     if os.path.exists(csv_foldername):
#
#         csv_filename = csv_foldername + "/eval_data.csv"
#
#         if os.path.exists(csv_filename):
#
#             df = pd.read_csv(csv_filename)
#
#             pre_avg = df.iloc[991:1001, 6].mean()
#             post_avg = df.iloc[1001:1011, 6].mean()
#             partial_avg = df.iloc[801:811, 6].mean()
#             final_avg = df.iloc[-10:, 6].mean()
#
#             pre.append(pre_avg)
#             post.append(post_avg)
#             partial.append(partial_avg)
#             final.append(final_avg)
#
#             dfs1.append(df)
#
#         else:
#
#             print(colored(csv_filename + " file is missing from the plot", "red"))
#
#     else:
#
#         print(colored(csv_foldername + " folder is missing from the plot", "red"))
#
# pre = pd.DataFrame(pre)
# post = pd.DataFrame(post)
# partial = pd.DataFrame(partial)
# final = pd.DataFrame(final)
#
# post_diff = (post - pre)
# post_diff_mean = round(post_diff.mean()[0], 4)
# post_diff_sem = round(post_diff.sem()[0], 4)
# post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
# post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
# post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)
#
# partial_diff = (partial - pre)
# partial_diff_mean = round(partial_diff.mean()[0], 4)
# partial_diff_sem = round(partial_diff.sem()[0], 4)
# partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
# partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
# partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)
#
# final_diff = (final - pre)
# final_diff_mean = round(final_diff.mean()[0], 4)
# final_diff_sem = round(final_diff.sem()[0], 4)
# final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
# final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
# final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)
#
# file.write(EXPERIMENT + "\n")
# file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
# file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
# file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))
#
# df1 = pd.concat(dfs1)
# df1 = df1.groupby(df1.index)
#
# df1_median = df1.median().iloc[::STEP, :]
# df1_std = df1.std().iloc[::STEP, :]
#
# x1 = df1_median.iloc[:, 1].to_numpy()
# median1 = df1_median.iloc[:, -1].to_numpy()
# std1 = df1_std.iloc[:, -1].to_numpy()
# sem1 = std1 / np.sqrt(30)
#
# lower1 = median1 - 2.045 * sem1
# upper1 = median1 + 2.045 * sem1
#
# # PPO network reinitialization
#
# EXPERIMENT = "PPOv2_AntEnv-v5:200000000_lr:0.00025_lrd:True_g:0.98_ns:512_mbs:256_epo:4_eps:0.1_c1:0.5_c2:0.01_cvl:False_mgn:0.5_gae:True_lam:0.95_hd:64_lstd:0.0_tef:200000_ee:10_tmsf:10000000_d:cpu_resumed"
#
# dfs2 = []
#
# post = []  # after a fault
# partial = []  # after partial adaptation (2 million time steps)
# final = []  # after learning is complete
#
# for seed in range(RUNS):
#
#     seed_data_dir = DATA_DIR + EXPERIMENT + "/seed{}".format(seed)
#     csv_foldername = seed_data_dir + "/csv"
#
#     if os.path.exists(csv_foldername):
#
#         csv_filename = csv_foldername + "/eval_data.csv"
#
#         if os.path.exists(csv_filename):
#
#             df = pd.read_csv(csv_filename)
#
#             post_avg = df.iloc[:10, 6].mean()
#             partial_avg = df.iloc[200:210, 6].mean()
#             final_avg = df.iloc[-10:, 6].mean()
#
#             post.append(post_avg)
#             partial.append(partial_avg)
#             final.append(final_avg)
#
#             dfs2.append(df)
#
#         else:
#
#             print(colored(csv_filename + " file is missing from the plot", "red"))
#
#     else:
#
#         print(colored(csv_foldername + " folder is missing from the plot", "red"))
#
# post = pd.DataFrame(post)
# partial = pd.DataFrame(partial)
# final = pd.DataFrame(final)
#
# post_diff = (post - pre)
# post_diff_mean = round(post_diff.mean()[0], 4)
# post_diff_sem = round(post_diff.sem()[0], 4)
# post_diff_ci = (post_diff.std() / np.sqrt(30))[0] * 2.045
# post_diff_ci_lower = round(post_diff_mean - post_diff_ci, 4)
# post_diff_ci_upper = round(post_diff_mean + post_diff_ci, 4)
#
# partial_diff = (partial - pre)
# partial_diff_mean = round(partial_diff.mean()[0], 4)
# partial_diff_sem = round(partial_diff.sem()[0], 4)
# partial_diff_ci = (partial_diff.std() / np.sqrt(30))[0] * 2.045
# partial_diff_ci_lower = round(partial_diff_mean - partial_diff_ci, 4)
# partial_diff_ci_upper = round(partial_diff_mean + partial_diff_ci, 4)
#
# final_diff = (final - pre)
# final_diff_mean = round(final_diff.mean()[0], 4)
# final_diff_sem = round(final_diff.sem()[0], 4)
# final_diff_ci = (final_diff.std() / np.sqrt(30))[0] * 2.045
# final_diff_ci_lower = round(final_diff_mean - final_diff_ci, 4)
# final_diff_ci_upper = round(final_diff_mean + final_diff_ci, 4)
#
# file.write(EXPERIMENT + "\n")
# file.write("POST {} & {} & {} & {}\n".format(post_diff_mean, post_diff_sem, post_diff_ci_lower, post_diff_ci_upper))
# file.write("PARTIAL {} & {} & {} & {}\n".format(partial_diff_mean, partial_diff_sem, partial_diff_ci_lower, partial_diff_ci_upper))
# file.write("FINAL {} & {} & {} & {}\n".format(final_diff_mean, final_diff_sem, final_diff_ci_lower, final_diff_ci_upper))
#
# df2 = pd.concat(dfs2)
# df2 = df2.groupby(df2.index)
#
# df2_median = df2.median().iloc[::STEP, :]
# df2_std = df2.std().iloc[::STEP, :]
#
# x2 = df2_median.iloc[:, 1].to_numpy()
# median2 = df2_median.iloc[:, -1].to_numpy()
# std2 = df2_std.iloc[:, -1].to_numpy()
# sem2 = std2 / np.sqrt(20)
#
# lower2 = median2 - 2.045 * sem2
# upper2 = median2 + 2.045 * sem2
#
# fig = plt.figure()
# ax = plt.plot(x1, median1, color="blue", label="models")
# plt.fill_between(x1, lower1, upper1, color="blue", alpha=0.2)
# ax2 = plt.plot(x2+200000000, median2, color="green", label="nothing")
# plt.fill_between(x2+200000000, lower2, upper2, color="green", alpha=0.2)
# plt.axvline(x=200000000, alpha=1.0, color="red", ymax=0.05)
# plt.legend(loc="lower right")
# plt.ylim(-1000, 9000)
# pss.plot_settings()
# plt.xlabel("time steps")
# plt.ylabel("average\nreturn\n(30 runs)", rotation="horizontal", labelpad=30)
#
# plt.savefig(DATA_DIR + "images/v5_ppo_all2.jpg".format(EXPERIMENT))
# # plt.show()
# plt.close()

file.close()
