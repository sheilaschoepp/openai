import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description="Draw results of the experiments inside a directory")

parser.add_argument("-p", "--path", default="",
                    help="absolute path of the folder containing experiments")

args = parser.parse_args()


# PATH = '/home/mehran/Desktop/AntAnalyze'

def draw():
    PATH = args.path
    experiments_list = os.listdir(PATH)
    experiment_seed = {}

    # set the theme for plots
    sns.set_style("dark")
    sns.set_theme()

    for e in experiments_list:
        seed_list = os.listdir(os.path.join(PATH, e))
        experiment_seed[e] = seed_list

    for exp in experiments_list:
        num_seeds = len(experiment_seed[exp])
        path = os.path.join(PATH, exp, 'seed0', 'csv', 'eval_data.csv')
        data_temp = pd.read_csv(path)
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

        x = np.array(data_temp['num_time_steps'])[:-1]
        plt.plot(x, average, 'b')
        plt.fill_between(x, average - standard_error, average + standard_error, color='r', alpha=0.2)
        if not os.path.exists(os.path.join(PATH, 'draw_results')):
            os.mkdir(os.path.join(PATH, 'draw_results'))
        result_path = os.path.join(PATH, 'draw_results')
        plt.savefig(os.path.join(result_path, f'{exp}.jpg'), dpi=300)


if __name__ == "__main__":
    draw()
