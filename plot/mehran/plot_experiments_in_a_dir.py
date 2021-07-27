import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pathlib
import os, shutil, pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument("-d", "--dir", default="",
                    help="absolute path of the folder containing experiments")

parser.add_argument("-p", "--percentage_consider", default=1.0,
                    help="the sum of expected reward from the beginning to the percentage of time from the"
                         "whole experiment time that we consider")

parser.add_argument("-nte", "--num_top_experiments", default=10,
                    help="how many best settings do you want to have?")

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


def draw():
    PATH = args.dir
    percentage_consider = float(args.percentage_consider)
    num_top_experiments = int(args.num_top_experiments)
    experiments_list = os.listdir(PATH)

    current_path = pathlib.Path(__file__).parent.absolute()
    if not os.path.exists(os.path.join(current_path, 'plotted_hps_results')):
        os.mkdir(os.path.join(current_path, 'plotted_hps_results'))
    result_path = os.path.join(current_path, 'plotted_hps_results')

    if not os.path.exists(os.path.join(result_path, 'best_hps_results')):
        os.mkdir(os.path.join(result_path, 'best_hps_results'))
    best_hps_results_path = os.path.join(result_path, 'best_hps_results')

    experiment_seed = {}
    experiments_score_sac = {}
    experiments_score_ppo = {}
    experiments_statistical_info = {}
    # set the theme for plots
    sns.set_style("dark")
    sns.set_theme()
    e = 0

    # List the experiments in a directory
    while e < len(experiments_list):
        try:
            seed_list = os.listdir(os.path.join(PATH, experiments_list[e]))
            experiment_seed[experiments_list[e]] = seed_list
            e += 1
        except PermissionError:
            del experiments_list[e]
            continue

    # Calculate the average and standard error of each experiment and
    # draw results for each one
    with tqdm(total=len(experiments_list)) as pbar:
        for exp in experiments_list:
            pss = find_pss(exp)
            num_seeds = len(experiment_seed[exp])
            try:
                path = os.path.join(PATH, exp, 'seed0', 'csv', 'eval_data.csv')
                data_temp = pd.read_csv(path)
            except FileNotFoundError:
                continue

            num_samples = len(data_temp) - 1
            average_returns = np.zeros([num_seeds, num_samples])
            index = 0
            for seed in experiment_seed[exp]:
                path = os.path.join(PATH, exp, seed, 'csv', 'eval_data.csv')
                try:
                    data_temp = pd.read_csv(path)
                except FileNotFoundError:
                    continue
                average_returns[index] = np.array(data_temp['average_return'])[:-1]
                index += 1

            average = np.mean(average_returns, axis=0)
            standard_error = np.std(average_returns, axis=0) / np.sqrt(average_returns.shape[0])

            if 'SAC' in exp:
                experiments_score_sac[exp] = average[:int(average.shape[0] * percentage_consider)].sum()
            elif 'PPO' in exp:
                experiments_score_ppo[exp] = average[:int(average.shape[0] * percentage_consider)].sum()

            x = np.array(data_temp['num_time_steps'])[:-1]
            experiments_statistical_info[exp] = {'avg': average, 'std_error': standard_error, 'time_step': x,
                                                 'pss': pss}
            plt.figure(figsize=(12, 5))
            plt.plot(x, average, 'b')
            plt.fill_between(x, average - 2.26 * standard_error, average + 2.26 * standard_error, alpha=0.2)
            plt.savefig(os.path.join(result_path, f'{exp}.jpg'), dpi=300)
            plt.close()
            pbar.update(1)

    # Finding the best HP settings based on the sum of average returns of different seeds
    sorted_experiments_score_sac = {k: v for k, v in sorted(experiments_score_sac.items(), key=lambda item: item[1])}
    sorted_experiments_score_ppo = {k: v for k, v in sorted(experiments_score_ppo.items(), key=lambda item: item[1])}

    counter = 0
    plt.figure(figsize=(12, 5))

    for exp in reversed(sorted_experiments_score_sac.keys()):
        # draw_path = os.path.join(result_path, f'{exp}.jpg')
        # shutil.copy2(draw_path, best_hps_results_path)
        single_plot(exp, experiments_statistical_info)
        counter += 1
        if counter == num_top_experiments:
            break

    plt.legend(loc="upper right")
    plt.savefig(os.path.join(best_hps_results_path, f'best_results_sac.jpg'), dpi=300)
    plt.close()

    counter = 0
    plt.figure(figsize=(12, 5))

    for exp in reversed(sorted_experiments_score_ppo.keys()):
        # draw_path = os.path.join(result_path, f'{exp}.jpg')
        # shutil.copy2(draw_path, best_hps_results_path)
        single_plot(exp, experiments_statistical_info)
        counter += 1
        if counter == num_top_experiments:
            break

    plt.legend(loc="upper right")
    plt.savefig(os.path.join(best_hps_results_path, f'best_results_ppo.jpg'), dpi=300)
    plt.close()


def single_plot(exp, experiments_statistical_info):
    average = experiments_statistical_info[exp]['avg']
    standard_error = experiments_statistical_info[exp]['std_error']
    x = experiments_statistical_info[exp]['time_step']
    label = experiments_statistical_info[exp]['pss']
    plt.plot(x, average, label=label)
    plt.fill_between(x, average - 2.26 * standard_error, average + 2.26 * standard_error, alpha=0.2)


if __name__ == "__main__":
    draw()
