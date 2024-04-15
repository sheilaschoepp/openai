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


def main():

    # Data directory for Ant at 300k time steps.
    ant_data_dir = f"{os.getenv('HOME')}/Documents/openai/data/ant/exps/300k"

    # For comparison, the number of time steps in the ant fault
    # environment.
    ant_fault_time_steps = 300000

    # Cycle through ppo/sac and v1/v2/v3/v4 to compute post-fault
    # performances.
    for algo in os.listdir(ant_data_dir):
        algo_dir = os.path.join(ant_data_dir, algo)
        for version in os.listdir(algo_dir):
            version_dir = os.path.join(algo_dir, version)
            for exp in os.listdir(version_dir):
                exp_dir = os.path.join(version_dir, exp)
                compute_postfault_performance(exp_dir)


def compute_postfault_performance(dir_):

    pre = []
    post = []
    asymp = []

    for seed in range(0, 30):
        directory = os.path.join(dir_, "seed" + str(seed))
        if os.path.exists(directory):
            eval_data_dir = os.path.join(directory, "csv", "eval_data.csv")
            eval_data = pd.read_csv(eval_data_dir)
            pre.append(eval_data[199:200]["average_return"].values.tolist())
            normal_time_steps = 600000000 if "ppo" in directory else 20000000
            fault_time_steps = normal_time_steps + 300000
            index = eval_data[eval_data['num_time_steps'] == 'x'].index[0]
            post.append()
            # post.append(eval_data[postfault_min:postfault_max]["average_return"].values.tolist())
            # asymp.append(eval_data[-10:]["average_return"].values.tolist())
        else:
            print(colored("missing: " + directory, "red"))


"""
Main Function (Entry Point)
"""

if __name__ == "__main__":

    main()
