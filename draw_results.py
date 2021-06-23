import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import pathlib
import os, shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument("-d", "--dir", default="",
                    help="absolute path of the folder containing experiments")

parser.add_argument("-p", "--percentage_consider", default=0.2,
                    help="the sum of expected reward from the beginning to the percentage of time from the"
                         "whole experiment time that we consider")

parser.add_argument("-nte", "--num_top_experiments", default=5,
                    help="how many best settings do you want to have?")

args = parser.parse_args()


# PATH = '/home/mehran/Desktop/AntAnalyze'

def draw():
    PATH = args.dir
    percentage_consider = args.percentage_consider
    num_top_experiments = args.num_top_experiments
    experiments_list = os.listdir(PATH)

    current_path = pathlib.Path(__file__).parent.absolute()
    if not os.path.exists(os.path.join(current_path, 'draw_results')):
        os.mkdir(os.path.join(current_path, 'draw_results'))
    result_path = os.path.join(current_path, 'draw_results')

    if not os.path.exists(os.path.join(current_path, 'best_results')):
        os.mkdir(os.path.join(current_path, 'best_results'))
    best_results_path = os.path.join(current_path, 'best_results')

    experiment_seed = {}
    experiments_score = {}
    # set the theme for plots
    sns.set_style("dark")
    sns.set_theme()
    e = 0
    while e < len(experiments_list):
        try:
            seed_list = os.listdir(os.path.join(PATH, experiments_list[e]))
            experiment_seed[experiments_list[e]] = seed_list
            e += 1
        except PermissionError:
            del experiments_list[e]
            continue

    with tqdm(total=len(experiments_list)) as pbar:
        for exp in experiments_list:
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
                data_temp = pd.read_csv(path)
                average_returns[index] = np.array(data_temp['average_return'])[:-1]
                index += 1

            average = np.mean(average_returns, axis=0)
            standard_error = np.std(average_returns, axis=0) / np.sqrt(average_returns.shape[0])
            experiments_score[exp] = average[:int(average.shape[0] * percentage_consider)].sum()

            x = np.array(data_temp['num_time_steps'])[:-1]
            plt.figure(figsize=(12, 5))
            plt.plot(x, average, 'b')
            plt.fill_between(x, average - standard_error, average + standard_error, color='r', alpha=0.2)
            plt.savefig(os.path.join(result_path, f'{exp}.jpg'), dpi=300)
            pbar.update(1)

    sorted_experiments_score = {k: v for k, v in sorted(experiments_score.items(), key=lambda item: item[1])}

    counter = 0
    for exp in sorted_experiments_score.keys():
        print(exp)
        draw_path = os.path.join(result_path, f'{exp}.jpg')
        shutil.copy2(draw_path, best_results_path)
        counter += 1
        print(counter)
        print(num_top_experiments)
        print(counter == num_top_experiments)
        if counter == num_top_experiments:
            print('fuck yeah')
            break


if __name__ == "__main__":
    draw()
