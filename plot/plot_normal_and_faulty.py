import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pathlib
import os, shutil, pickle
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument("-nd", "--normaldir", default="",
                    help="absolute path of the folder containing the seeds of the normal environment")

parser.add_argument("-fd", "--faultydir", default="",
                    help="absolute path of the folder containing the experiments (not seeds) of the faulty environment "
                         "whose normal environment was the path provided in --normaldir")

args = parser.parse_args()


def find_pss(exp_dir):
    """
    Finds pss from the name of the file.
    """
    params = str.split(exp_dir, '_')
    pss = 0
    for p in reversed(params):
        if 'pss' in p:
            pss = str.split(p, ':')[-1]
    return pss


def find_env_name(exp_dir):
    return exp_dir.split('_')[1].split(':')[0]


def extract_params(exp_dir):
    return dict([(t.split(':') if len(t.split(':')) == 2 else [t.split(':')[0], None]) for t in exp_dir.split('_')])


def find_label(params):
    rn = params['rn']
    cm = params['cm']
    label = ''
    label += 'rn' if rn == "True" else ''
    label += 'cm' if cm == 'True' else ''
    return label if label != '' else 'keep_both'


X_AXIS = ['num_time_steps', 'num_updates', 'num_samples']
X_AXIS_TO_LABEL = {'num_time_steps': 'Time step',
                   'num_updates': 'Number of updates',
                   'num_samples': 'Number of samples'}


def draw():
    NORMAL_PATH = args.normaldir
    FAULTY_PATH = args.faultydir
    faulty_experiments_list = os.listdir(FAULTY_PATH)
    env_name = find_env_name(faulty_experiments_list[0])

    current_path = pathlib.Path(__file__).parent.absolute()
    result_path = os.path.join(current_path, 'normal_faulty_plots', env_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    experiment_seed = {}
    experiments_statistical_info = {}
    # set the theme for plots
    sns.set_style("dark")
    sns.set_theme()

    normal_seed_list = os.listdir(NORMAL_PATH)
    experiment_seed['normal'] = normal_seed_list

    # List the experiments in a directory
    for exp in faulty_experiments_list:
        seed_list = os.listdir(os.path.join(FAULTY_PATH, exp))
        experiment_seed[exp] = seed_list

    # Calculate the average and standard error of each experiment and
    # draw results for each one
    for exp in experiment_seed.keys():
        num_seeds = len(experiment_seed[exp])
        average_returns = None
        data_temp = None

        index = 0
        for seed in experiment_seed[exp]:
            path = os.path.join(FAULTY_PATH, exp, seed, 'csv', 'eval_data.csv') if exp != 'normal' else \
                os.path.join(NORMAL_PATH, seed, 'csv', 'eval_data.csv')
            try:
                data_temp = pd.read_csv(path)
            except FileNotFoundError:
                continue
            if average_returns is None:
                average_returns = np.zeros([num_seeds, len(data_temp) - 1])

            average_returns[index] = np.array(data_temp['average_return'])[:-1]
            index += 1

        average = np.mean(average_returns, axis=0)
        standard_error = np.std(average_returns, axis=0) / np.sqrt(average_returns.shape[0])

        if exp == 'normal':
            start_index = 0
        else:
            start_index = experiments_statistical_info['normal']['avg'].shape[0]

        num_time_steps = np.array(data_temp['num_time_steps'])[start_index:-2]
        num_updates = np.array(data_temp['num_updates'])[start_index:-2]
        num_samples = np.array(data_temp['num_samples'])[start_index:-2]

        # In the faulty environments when we throw away the memory or network, number of updates and samples would
        # become zero. These lines are for making the plots consistent for number of samples and updates
        if exp != 'normal' and np.any(num_updates == 0):
            index = np.where(num_updates == 0)[0][0]
            num_updates[index:] += num_updates[index - 1]
        if exp != 'normal' and np.any(num_samples == 0):
            index = np.where(num_samples == 0)[0][0]
            num_samples[index:] += num_samples[index - 1]

        experiments_statistical_info[exp] = {'avg': average[start_index: -1],
                                             'std_error': standard_error[start_index: -1],
                                             'num_time_steps': num_time_steps,
                                             'num_updates': num_updates,
                                             'num_samples': num_samples}

    linestyles = ['-', '--', '-.', ':']
    markers = ['*', 'h']

    for x in X_AXIS:
        # Create main container with size of 6x5
        fig = plt.figure(figsize=(12, 7))
        plt.subplots_adjust(bottom=0.1, left=0.1, top=.9, right=.9)

        # Create first axes, the top-left plot with green plot
        sub1 = fig.add_subplot(2, 4, (2, 3))  # two rows, two columns, second cell

        # Create second axes, the top-left plot with orange plot
        sub2 = fig.add_subplot(2, 1, 2)  # two rows, two columns, second cell

        magnify_interval_length = 20
        # This parameter is for controlling the magnifying interval is  only plotted once
        already_filled_interval = False
        color_index = 0
        marker_index = 0

        min_ylim, max_ylim = -5, 0

        for exp in experiments_statistical_info:
            label = 'normal' if exp == 'normal' else find_label(extract_params(exp))
            x_values = experiments_statistical_info[exp][x]
            average = experiments_statistical_info[exp]['avg']
            standard_error = experiments_statistical_info[exp]['std_error']

            # TODO: change ylim to be dynamic according to the results

            tmp = sub2.plot(x_values, average, label=label)[0]
            color = tmp.get_color()
            sub2.fill_between(x_values, average - 2.26 * standard_error, average + 2.26 * standard_error,
                              alpha=0.2, color=color)
            if exp != 'normal':
                if color_index < len(linestyles):
                    sub1.plot(x_values, average, linestyle=linestyles[color_index], color=color, label=label)
                    sub1.fill_between(x_values, average - 2.26 * standard_error, average + 2.26 * standard_error,
                                      alpha=0.2, color=color)
                    sub1.set_xlim(x_values[0], x_values[magnify_interval_length])
                    sub1.set_ylim(min_ylim, max_ylim)
                    sub1.set_ylabel('y', labelpad=15)
                    color_index += 1
                else:
                    sub1.plot(x_values, average, marker=markers[marker_index], color=color, label=label)
                    sub1.fill_between(x_values, average - 2.26 * standard_error, average + 2.26 * standard_error,
                                      alpha=0.2, color=color)
                    sub1.set_xlim(x_values[0], x_values[magnify_interval_length])
                    sub1.set_ylim(min_ylim, max_ylim)
                    sub1.set_ylabel('y', labelpad=15)
                    marker_index += 1

            if exp != 'normal' and not already_filled_interval:
                already_filled_interval = True
                # Create blocked area in third axes
                sub2.fill_between((x_values[0], x_values[magnify_interval_length]), min_ylim, max_ylim,
                                  facecolor='black', alpha=0.2)  # blocked area for first axes

                # TODO: xyB=(x, ylim) change the ylim here when you changed the ylim above
                # Create left side of Connection patch for first axes

                con1 = ConnectionPatch(xyA=(x_values[0], min_ylim / 2), coordsA=sub2.transData,
                                       xyB=(x_values[0], min_ylim), coordsB=sub1.transData, color='black')
                # Add left side to the figure
                fig.add_artist(con1)

                # Create right side of Connection patch for first axes
                con2 = ConnectionPatch(xyA=(x_values[magnify_interval_length], min_ylim / 2), coordsA=sub2.transData,
                                       xyB=(x_values[magnify_interval_length], min_ylim), coordsB=sub1.transData,
                                       color='black')
                # Add right side to the figure
                fig.add_artist(con2)

        handles, labels = sub1.get_legend_handles_labels()
        handles2, labels2 = sub2.get_legend_handles_labels()
        labels = ['normal'] + labels
        handles = [handles2[labels2.index('normal')]] + handles
        fig.legend(handles, labels, loc='upper right')
        sub2.set_xlabel(X_AXIS_TO_LABEL[x])
        sub2.set_ylabel('Episode Cumulative Reward')

        # plt.title(x)
        plt.savefig(os.path.join(result_path, f'x_axis_{x}.jpg'), dpi=300)
        # plt.close()


def single_plot(exp, experiments_statistical_info):
    average = experiments_statistical_info[exp]['avg']
    standard_error = experiments_statistical_info[exp]['std_error']
    x = experiments_statistical_info[exp]['time_step']
    label = experiments_statistical_info[exp]['pss']
    plt.plot(x, average, label=label)
    plt.fill_between(x, average - 2.26 * standard_error, average + 2.26 * standard_error, alpha=0.2)


if __name__ == "__main__":
    draw()
