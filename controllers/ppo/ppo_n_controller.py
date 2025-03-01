import argparse
import csv
import itertools
import matplotlib.pyplot as plt
import optuna
import os

# The following three lines must come before numpy import.
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import pandas as pd
import pickle
import random
import sys
import time
import torch
import wandb

from copy import copy, deepcopy
from datetime import date, timedelta
from os import path
from shutil import rmtree
from termcolor import colored

import utils.plot_style_settings as pss
from controllers.ppo.ppo_agent import PPO
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs  # do not delete; required for custom gym environments

parser = argparse.ArgumentParser(description='PyTorch Proximal Policy Optimization Arguments')

parser.add_argument('-e', '--n_env_name', default='Ant-v5',  # Ant-v5 or FetchReachDense-v3
                    help='name of normal (non-malfunctioning) MuJoCo Gym environment (default: Ant-v5)')
parser.add_argument('-t', '--n_time_steps', type=int, default=10000000, metavar='N',  # Ant-v5: 10000000 or FetchReachDense-v3: 100000
                    help='number of time steps in normal (non-malfunctioning) MuJoCo Gym environment (default: 10000000)')

parser.add_argument('--lr', type=float, default=0.000275, metavar='G',
                    help='learning rate (default: 0.000275)')
parser.add_argument('-lrd', '--linear_lr_decay', default=False, action='store_true',
                    help='if true, decrease learning rate linearly (default: False)')
parser.add_argument('--gamma', type=float, default=0.848, metavar='G',
                    help='discount factor (default: 0.848)')

parser.add_argument('-ns', '--num_samples', type=int, default=3424, metavar='N',
                    help='number of samples used to update the network(s) (default: 3424)')
parser.add_argument('-mbs', '--mini_batch_size', type=int, default=8, metavar='N',
                    help=' number of samples per mini-batch (default: 8)')
parser.add_argument('--epochs', type=int, default=24, metavar='N',
                    help='number of epochs when updating the network(s) (default: 24)')

parser.add_argument('--epsilon', type=float, default=0.3, metavar='G',
                    help='clip parameter (default: 0.3)')
parser.add_argument('--vf_loss_coef', type=float, default=1.0, metavar='G',
                    help=' c1 - coefficient for the squared error loss term (default: 1.0)')
parser.add_argument('--policy_entropy_coef', type=float, default=0.0007, metavar='G',
                    help=' c2 - coefficient for the entropy bonus term (default: 0.0007)')
parser.add_argument('--clipped_value_fn', default=False, action='store_true',
                    help='if true, clip value function (default: False)')
parser.add_argument('--max_grad_norm', type=float, default=0.5, metavar='G',
                    help=' max norm of gradients (default: 0.5)')

parser.add_argument('--use_gae', default=False, action='store_true',
                    help=' if true, use generalized advantage estimation (default: False)')
parser.add_argument('--gae_lambda', type=float, default=0.9327, metavar='G',
                    help='generalized advantage estimation smoothing parameter (default: 0.9327)')

parser.add_argument('-nr', '--normalize_rewards', default=False, action='store_true',
                    help='if true, normalize rewards in memory (default: False)')

parser.add_argument('--hidden_dim', type=int, default=64, metavar='N',
                    help='hidden dimension (default: 64)')
parser.add_argument('--log_std', type=float, default=0.0, metavar='G',
                    help='log standard deviation of the policy distribution (default: 0.0)')

parser.add_argument('-tef', '--time_step_eval_frequency', type=int, default=50000, metavar='N',  # Ant-v5: 50000 or FetchReachDense-v3: 500
                    help='frequency of policy evaluation during learning (default: 50000)')
parser.add_argument('-ee', '--eval_episodes', type=int, default=10, metavar='N',
                    help='number of episodes in policy evaluation roll-out (default: 10)')

parser.add_argument('-c', '--cuda', default=False, action='store_true',
                    help='if true, run on GPU (default: False)')

parser.add_argument('-s', '--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')

parser.add_argument('-d', '--delete', default=False, action='store_true',
                    help='if true, delete previously saved data and restart training (default: False)')

parser.add_argument('-wb', '--wandb', default=False, action='store_true',
                    help='if true, log to weights & biases (default: False)')

parser.add_argument('-o', '--optuna', default=False, action='store_true',
                    help='if true, run a parameter search with optuna (default: False)')

args = parser.parse_args()


class NormalController:
    """
    Controller for learning in the normal environment.

    The experiment program directs the experiment's execution, including the sequence of agent-environment interactions
    and agent performance evaluation.  -Brian Tanner & Adam White
    """

    LINE = '--------------------------------------------------------------------------------'

    def __init__(self):

        # experiment runtime information

        self.start = time.time()

        # experiment parameters

        self.parameters = None

        self.parameters = {
            'n_env_name': args.n_env_name,
            'n_time_steps': args.n_time_steps,
            'lr': args.lr,
            'linear_lr_decay': args.linear_lr_decay,
            'gamma': args.gamma,
            'num_samples': args.num_samples,
            'mini_batch_size': args.mini_batch_size,
            'epochs': args.epochs,
            'epsilon': args.epsilon,
            'vf_loss_coef': args.vf_loss_coef,
            'policy_entropy_coef': args.policy_entropy_coef,
            'clipped_value_fn': args.clipped_value_fn,
            'max_grad_norm': args.max_grad_norm,
            'use_gae': args.use_gae,
            'gae_lambda': args.gae_lambda,
            'normalize_rewards': args.normalize_rewards,
            'hidden_dim': args.hidden_dim,
            'log_std': args.log_std,
            'time_step_eval_frequency': args.time_step_eval_frequency,
            'eval_episodes': args.eval_episodes,
            'cuda': args.cuda,
            'device': 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu',
            'seed': args.seed,
            'wandb': args.wandb,
            'optuna': args.optuna
        }

        # W&B initialization

        if self.parameters["wandb"]:

            env = self.parameters["n_env_name"].split('-')[0].lower()
            version = self.parameters["n_env_name"].split('-')[1].lower()
            prefix = f'{env}{version}'

            wandb.init(
                project=f'{prefix}_ppo',
                config=self.parameters,
                dir=f'{os.getenv("HOME")}/Documents/openai'
            )

            wandb.define_metric(
                name='Key Metrics/*',
                step_metric='Time Steps'
            )

            wandb.define_metric(
                name='Loss Metrics/*',
                step_metric='Number of Updates'
            )

        # experiment data directory

        suffix = (
            f'{self.parameters["n_env_name"]}:{self.parameters["n_time_steps"]}'
            f'_lr:{self.parameters["lr"]}'
            f'_lrd:{self.parameters["linear_lr_decay"]}'
            f'_g:{self.parameters["gamma"]}'
            f'_ns:{self.parameters["num_samples"]}'
            f'_mbs:{self.parameters["mini_batch_size"]}'
            f'_epo:{self.parameters["epochs"]}'
            f'_eps:{self.parameters["epsilon"]}'
            f'_c1:{self.parameters["vf_loss_coef"]}'
            f'_c2:{self.parameters["policy_entropy_coef"]}'
            f'_cvf:{self.parameters["clipped_value_fn"]}'
            f'_mgn:{self.parameters["max_grad_norm"]}'
            f'_gae:{self.parameters["use_gae"]}'
            f'_lam:{self.parameters["gae_lambda"]}'
            f'_nr:{self.parameters["normalize_rewards"]}'
            f'_hd:{self.parameters["hidden_dim"]}'
            f'_lstd:{self.parameters["log_std"]}'
            f'_tef:{self.parameters["time_step_eval_frequency"]}'
            f'_ee:{self.parameters["eval_episodes"]}'
            f'_d:{self.parameters["device"]}'
            f'{"_wb" if self.parameters["wandb"] else ""}'
            f'{"_o" if self.parameters["optuna"] else ""}'
        )

        self.experiment = f'PPO_{suffix}'

        self.data_dir = f'{os.getenv("HOME")}/Documents/openai/data/{self.experiment}/seed{self.parameters["seed"]}'

        # are we restarting training?  do the data files for the selected seed already exist?
        if path.exists(self.data_dir):

            print(self.LINE)
            print(self.LINE)

            if args.delete:
                # yes; argument flag present to indicate data deletion
                print(colored(text='argument indicates DATA DELETION', color='red'))
                print(colored(text='deleting data...', color='red'))
                rmtree(path=self.data_dir, ignore_errors=True)
                print(colored(text='data deletion complete', color='red'))
            else:
                # yes; argument flag not present; get confirmation of data deletion from user input
                print(colored(text='You are about to delete saved data and restart training.', color='red'))
                s = input(colored(text='Are you sure you want to continue? Hit "y" then "Enter" to continue.\n', color='red'))
                if s == 'y':
                    # delete old data; rewrite new data to same location
                    print(colored(text='user input indicates DATA DELETION', color='red'))
                    print(colored(text='deleting data...', color='red'))
                    rmtree(path=self.data_dir, ignore_errors=True)
                    print(colored(text='data deletion complete', color='red'))
                else:
                    # do not delete old data; system exit
                    print(colored(text='user input indicates NO DATA DELETION', color='red'))
                    print(self.LINE)
                    sys.exit('\nexiting...')

        # data

        num_rows = int(self.parameters["n_time_steps"] / self.parameters["time_step_eval_frequency"]) + 1  # add 1 for evaluation before any learning (0th entry)
        num_columns = 3
        self.eval_data = np.zeros((num_rows, num_columns))

        num_rows = (self.parameters["n_time_steps"] // self.parameters["num_samples"])
        num_columns = 8
        self.loss_data = np.zeros((num_rows, num_columns))

        # seeds

        random.seed(self.parameters["seed"])
        np.random.seed(self.parameters["seed"])
        torch.manual_seed(self.parameters["seed"])
        if self.parameters["device"] == 'cuda':
            torch.cuda.manual_seed(self.parameters["seed"])

        # env is seeded in Environment env_init() method

        # rl problem

        # environment used for training
        self.env = Environment(self.parameters["n_env_name"],
                               self.parameters["seed"])

        # environment used for evaluation
        self.eval_env = Environment(self.parameters["n_env_name"],
                                    self.parameters["seed"])

        # agent
        self.agent = PPO(self.env.env_observation_dim(),
                         self.env.env_action_dim(),
                         self.parameters["hidden_dim"],
                         self.parameters["log_std"],
                         self.parameters["lr"],
                         self.parameters["linear_lr_decay"],
                         self.parameters["gamma"],
                         self.parameters["n_time_steps"],
                         self.parameters["num_samples"],
                         self.parameters["mini_batch_size"],
                         self.parameters["epochs"],
                         self.parameters["epsilon"],
                         self.parameters["vf_loss_coef"],
                         self.parameters["policy_entropy_coef"],
                         self.parameters["clipped_value_fn"],
                         self.parameters["max_grad_norm"],
                         self.parameters["use_gae"],
                         self.parameters["gae_lambda"],
                         self.parameters["normalize_rewards"],
                         self.parameters["device"],
                         self.parameters["wandb"],
                         self.loss_data)

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = {'num_episodes': 0,
                               'num_steps': 0,
                               'total_reward': 0}

        # print summary info

        # is GPU being used?

        print(self.LINE)

        if self.parameters["device"] == 'cuda':
            print('NOTE: GPU is being used for this experiment.')
        else:
            print('NOTE: GPU is not being used for this experiment.  CPU only.')

        # what are the experiment parameters?

        print(self.LINE)

        def highlight_non_default_values(argument):
            """
            Highlight non-default argument values in printed summary.

            Note: Non-default values are printed in red.

            @param argument: str
                the argument name

            @return: str
                the value of the argument
            """
            default = parser.get_default(argument)
            if self.parameters[argument] != default:
                return colored(self.parameters[argument], 'red')
            else:
                return self.parameters[argument]

        print(f'normal environment name: {highlight_non_default_values("n_env_name")}')
        print(f'normal time steps: {highlight_non_default_values("n_time_steps")}')
        print(f'lr: {highlight_non_default_values("lr")}')
        print(f'linear lr decay: {highlight_non_default_values("linear_lr_decay")}')
        print(f'gamma: {highlight_non_default_values("gamma")}')
        print(f'number of samples: {highlight_non_default_values("num_samples")}')
        print(f'mini-batch size: {highlight_non_default_values("mini_batch_size")}')
        print(f'epochs: {highlight_non_default_values("epochs")}')
        print(f'epsilon: {highlight_non_default_values("epsilon")}')
        print(f'value function loss coefficient: {highlight_non_default_values("vf_loss_coef")}')
        print(f'policy entropy coefficient: {highlight_non_default_values("policy_entropy_coef")}')
        print(f'clipped value function: {highlight_non_default_values("clipped_value_fn")}')
        print(f'max norm of gradients: {highlight_non_default_values("max_grad_norm")}')
        print(f'use generalized advantage estimation: {highlight_non_default_values("use_gae")}')
        print(f'gae smoothing coefficient (lambda): {highlight_non_default_values("gae_lambda")}')
        print(f'normalize rewards: {highlight_non_default_values("normalize_rewards")}')
        print(f'hidden dimension: {highlight_non_default_values("hidden_dim")}')
        print(f'log_std: {highlight_non_default_values("log_std")}')
        print(f'time step evaluation frequency: {highlight_non_default_values("time_step_eval_frequency")}')
        print(f'evaluation episodes: {highlight_non_default_values("eval_episodes")}')

        if self.parameters["device"] == 'cuda':
            print(f'device: {self.parameters["device"]}')
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                print(f'cuda visible device(s): {colored(os.environ["CUDA_VISIBLE_DEVICES"], "red")}')
            else:
                print(colored('cuda visible device(s): N/A', 'red'))
        else:
            print(f'device: {colored(self.parameters["device"], "red")}')

        print(f'seed: {colored(self.parameters["seed"], "red")}')
        print(f'wandb: {highlight_non_default_values("wandb")}')
        print(f'optuna: {highlight_non_default_values("optuna")}')

        print(self.LINE)
        print(self.LINE)

    def run(self):
        """
        Run Experiment.
        """

        self.rlg.rl_init(total_reward=self.rlg_statistics["total_reward"],
                         num_steps=self.rlg_statistics["num_steps"],
                         num_episodes=self.rlg_statistics["num_episodes"])

        # save the agent model and evaluate the model before any learning
        self.rlg.rl_agent_message(f'save_model, {self.data_dir}, {0}')
        self.evaluate_model(self.rlg.num_steps())

        for _ in itertools.count(1):

            # episode time steps are limited to 1000 (set below)
            # this is used to ensure that once self.parameters["n_time_steps"] is reached, the experiment is terminated
            max_steps_this_episode = min(1000, self.parameters["n_time_steps"] - self.rlg.num_steps())

            # run an episode
            self.rlg.rl_start()

            terminal = False

            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, _, terminal, _ = self.rlg.rl_step()

                # save and evaluate the model every
                # 'self.parameters["time_step_eval_frequency"]' time
                # steps
                if self.rlg.num_steps() % self.parameters["time_step_eval_frequency"] == 0:
                    self.rlg.rl_agent_message(f'save_model, {self.data_dir}, {self.rlg.num_steps()}')
                    self.evaluate_model(self.rlg.num_steps())

            # learning complete
            if self.rlg.num_steps() == self.parameters["n_time_steps"]:
                break

        self.save()
        self.plot()
        self.cleanup()

    def cleanup(self):
        """
        Close environment.
        Compute runtime.
        Save file with run information.
        Close wandb.
        """

        self.rlg.rl_env_message('close')

        run_time = str(timedelta(seconds=time.time() - self.start))[:-7]

        print(f'time to complete one run: {run_time} h:m:s')
        print(self.LINE)

        text_file = open(f'{self.data_dir}/run_summary.txt', 'w')
        text_file.write(date.today().strftime('%m/%d/%y'))
        text_file.write(f'\n\nExperiment {self.experiment}/seed{self.parameters["seed"]} complete.\n\nTime to complete: {run_time} h:m:s')
        text_file.close()

        if self.parameters["wandb"]:

            wandb.finish()

    def evaluate_model(self, num_time_steps):
        """
        Evaluates the current (deterministic) policy.

        @param num_time_steps: int
            the number of time steps into training

        Note: There is no agent learning in this method.
        Note: The environment used for evaluation is different from the environment which is used for learning.  Both
        environments have the same OpenAI name and version; thus, they use the same OpenAI code.
        Note: The same seed is used to create the environment for each evaluation.
        Note: We use a copy of the agent in this method so that we don't lose the value of Agent class attributes such
        as state and action.
        """

        if self.parameters["eval_episodes"] != 0:

            eval_agent = copy(self.agent)

            eval_rlg = RLGlue(self.eval_env, eval_agent)

            eval_rlg.rl_agent_message('mode, eval')

            returns = []

            eval_rlg.rl_init()

            for e in range(self.parameters["eval_episodes"]):

                eval_rlg.rl_start()

                terminal = False

                max_steps_this_episode = 1000
                while not terminal and ((max_steps_this_episode <= 0) or (eval_rlg.num_ep_steps() < max_steps_this_episode)):
                    _, _, terminal, _ = eval_rlg.rl_step()

                returns.append(eval_rlg.episode_reward())

            average_return = np.average(returns)

            real_time = int(time.time() - self.start)

            index = num_time_steps // self.parameters["time_step_eval_frequency"]
            self.eval_data[index] = [num_time_steps,
                                     average_return,
                                     real_time]

            cumulative_average_return = self.eval_data[:, 1].sum()

            if self.parameters["wandb"]:
                wandb.log(data={
                    'Key Metrics/Average Return': average_return,
                    'Key Metrics/Cumulative Average Return': cumulative_average_return,
                    'Key Metrics/Real Time': real_time,
                    'Time Steps': num_time_steps
                })

            print(f'evaluation at {num_time_steps} time steps: {average_return}')

            run_time = str(timedelta(seconds=time.time() - self.start))[:-7]
            print(f'runtime: {run_time} h:m:s')
            print(self.LINE)

    def plot(self):
        """
        Plot experiment data.

        File format: .jpg
        """

        print('plotting...')

        csv_foldername = f'{self.data_dir}/csv'
        os.makedirs(csv_foldername, exist_ok=True)

        jpg_foldername = f'{self.data_dir}/jpg'
        os.makedirs(jpg_foldername, exist_ok=True)

        df = pd.read_csv(f'{csv_foldername}/eval_data.csv')

        # evaluation: average_return vs num_time_steps
        df.plot(x='num_time_steps', y='average_return', color='blue', legend=False)
        plt.xlabel('time_steps')
        plt.ylabel('average\nreturn', rotation='horizontal', labelpad=30)
        plt.title('Policy Evaluation')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/evaluation_time_steps.jpg')
        plt.close()

        df = pd.read_csv(f'{csv_foldername}/loss_data.csv')

        # training: clip_loss vs num_updates
        df.plot(x='num_updates', y='clip_loss', color='blue', legend=False)
        plt.xlabel('updates')
        plt.ylabel('loss', rotation='horizontal', labelpad=30)
        plt.title('CLIP Loss')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/clip_loss_updates.jpg')
        plt.close()

        # training: vf_loss vs num_updates
        df.plot(x='num_updates', y='vf_loss', color='blue', legend=False)
        plt.xlabel('updates')
        plt.ylabel('loss', rotation='horizontal', labelpad=30)
        plt.title('VF Loss')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/vf_loss_updates.jpg')
        plt.close()

        # training: entropy vs num_updates
        df.plot(x='num_updates', y='entropy', color='blue', legend=False)
        plt.xlabel('updates')
        plt.ylabel('entropy', rotation='horizontal', labelpad=30)
        plt.title('Entropy')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/entropy_updates.jpg')
        plt.close()

        # training: clip_vf_s_loss vs num_updates
        df.plot(x='num_updates', y='clip_vf_s_loss', color='blue', legend=False)
        plt.xlabel('updates')
        plt.ylabel('loss', rotation='horizontal', labelpad=30)
        plt.title('CLIP+VF+S Loss')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/clip_vf_s_loss_updates.jpg')
        plt.close()

        # training: clip_fraction vs num_updates
        df.plot(x='num_updates', y='clip_fraction', color='blue', legend=False)
        plt.xlabel('updates')
        plt.ylabel('clip fraction', rotation='horizontal', labelpad=30)
        plt.title('Clip Fraction')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/clip_fraction_updates.jpg')
        plt.close()

        print('plotting complete')

        print(self.LINE)

    def save(self):
        """
        Save experiment data before policy evaluation begins.

        Save experiment parameters.
        Save experiment data: train data and loss data.
        Save seed state: random, torch, and numpy seed states.
        Save rlg statistics: num_episodes, num_steps, and total_reward.
        Save environment data: OpenAI seed states.
        Save agent data: models, number of updates of models, and memory.
        """

        print('saving...')

        self.save_seed_state()

        self.save_parameters()

        self.save_data()

        # save rlg data
        self.save_rlg_statistics()

        # save environment data
        self.rlg.rl_env_message(f'save, {self.data_dir}')

        # save agent data
        self.rlg.rl_agent_message(f'save, {self.data_dir}, {self.rlg.num_steps()}')

        print('saving complete')

        print(self.LINE)

    def save_data(self):
        """
        Save experiment data.

        File format: .csv
        """

        csv_foldername = f'{self.data_dir}/csv'
        os.makedirs(csv_foldername, exist_ok=True)

        eval_data_df = pd.DataFrame({'num_time_steps': self.eval_data[:, 0],
                                     'average_return': self.eval_data[:, 1],
                                     'real_time': self.eval_data[:, 2]})
        eval_data_df.to_csv(f'{csv_foldername}/eval_data.csv', float_format='%f')

        loss_data_df = pd.DataFrame({'num_updates': self.loss_data[:, 0],
                                     'num_epoch_updates': self.loss_data[:, 1],
                                     'num_mini_batch_updates': self.loss_data[:, 2],
                                     'clip_loss': self.loss_data[:, 3],
                                     'vf_loss': self.loss_data[:, 4],
                                     'entropy': self.loss_data[:, 5],
                                     'clip_vf_s_loss': self.loss_data[:, 6],
                                     'clip_fraction': self.loss_data[:, 7]})
        loss_data_df.to_csv(f'{csv_foldername}/loss_data.csv', float_format='%f')

    def save_parameters(self):
        """
        Save experiment parameters.

        File format: .csv and .pickle
        """

        csv_foldername = f'{self.data_dir}/csv'
        os.makedirs(csv_foldername, exist_ok=True)

        csv_filename = f'{csv_foldername}/parameters.csv'

        f = open(csv_filename, 'w')
        writer = csv.writer(f)
        for key, val in self.parameters.items():
            writer.writerow([key, val])
        f.close()

        pickle_foldername = f'{self.data_dir}/pickle'
        os.makedirs(pickle_foldername, exist_ok=True)

        pickle_filename = f'{pickle_foldername}/parameters.pickle'

        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_rlg_statistics(self):
        """
        Save rlg statistics: num_episodes, num_steps, and total_reward.

        File format: .pickle
        """

        self.rlg_statistics["num_episodes"] = self.rlg.num_episodes()
        self.rlg_statistics["num_steps"] = self.rlg.num_steps()
        self.rlg_statistics["total_reward"] = self.rlg.total_reward()

        pickle_foldername = f'{self.data_dir}/pickle'
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(f'{pickle_foldername}/rlg_statistics.pickle', 'wb') as f:
            pickle.dump(self.rlg_statistics, f)

    def save_seed_state(self):
        """
        Save state of the random number generators used by random, numpy, and pytorch.

        File format: .pickle (random and numpy) and .pt (pytorch)
        """

        pickle_foldername = f'{self.data_dir}/pickle'
        os.makedirs(pickle_foldername, exist_ok=True)

        random_random_state = random.getstate()
        numpy_random_state = np.random.get_state()

        with open(f'{pickle_foldername}/random_random_state.pickle', 'wb') as f:
            pickle.dump(random_random_state, f)

        with open(f'{pickle_foldername}/numpy_random_state.pickle', 'wb') as f:
            pickle.dump(numpy_random_state, f)

        pt_foldername = f'{self.data_dir}/pt'
        os.makedirs(pt_foldername, exist_ok=True)

        torch_random_state = torch.get_rng_state()
        torch.save(torch_random_state, f'{pt_foldername}/torch_random_state.pt')

        if self.parameters["device"] == 'cuda':
            torch_cuda_random_state = torch.cuda.get_rng_state()
            torch.save(torch_cuda_random_state, f'{pt_foldername}/torch_cuda_random_state.pt')


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning of the PPO
    agent.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object that provides
        parameter suggestions.

    Returns:
        float: The average return after training for the specified
        number of time steps.
    """

    # Set the learning rate.
    lr = trial.suggest_float(name='lr',
                             low=0.00001,
                             high=0.001,
                             step=0.00000001)

    # Set the linear learning rate decay flag.
    linear_lr_decay_choices = [True, False]
    linear_lr_decay = trial.suggest_categorical(name='linear_lr_decay',
                                                choices=linear_lr_decay_choices)

    # Set gamma.
    gamma = trial.suggest_float(name='gamma',
                                low=0.8,
                                high=0.9999,
                                step=0.0001)

    # Set the number of samples.
    num_samples_choices = [1024, 2048, 4096, 8192]
    num_samples = trial.suggest_categorical(name='num_samples',
                                            choices=num_samples_choices)

    # Set the mini-batch size.
    mini_batch_size_choices = [32, 64, 128, 256, 512]
    mini_batch_size = trial.suggest_categorical(name='mini_batch_size',
                                                choices=mini_batch_size_choices)

    # If mini_batch_size is greater than num_samples, prune the trial.
    if mini_batch_size > num_samples:
        raise optuna.TrialPruned()

    # Set the number of epochs.
    epochs = trial.suggest_int(name='num_epochs',
                               low=3,
                               high=10)

    # Set epsilon.
    epsilon = trial.suggest_float(name='epsilon',
                                  low=0.1,
                                  high=0.3,
                                  step=0.0001)

    # Set the value function loss coefficient.
    vf_loss_coef = trial.suggest_float(name='vf_loss_coef',
                                       low=0.1,
                                       high=1.0,
                                       step=0.0001)

    # Set the policy entropy coefficient.
    policy_entropy_coef = trial.suggest_float(name='policy_entropy_coef',
                                              low=0.0001,
                                              high=0.1,
                                              step=0.0000001)

    # Set the clipped value function flag.
    clipped_value_fn_choices = [True, False]
    clipped_value_fn = trial.suggest_categorical(name='clipped_value_fn',
                                                 choices=clipped_value_fn_choices)

    # Set max grad norm.
    max_grad_norm_choices = [0.5, 1.0]
    max_grad_norm = trial.suggest_categorical(name='max_grad_norm',
                                              choices=max_grad_norm_choices)

    # Set the use gae flag.
    use_gae_choices = [True, False]
    use_gae = trial.suggest_categorical(name='use_gae',
                                        choices=use_gae_choices)

    # Set the GAE lambda parameter.
    gae_lambda = trial.suggest_float(name='gae_lambda',
                                     low=0.9,
                                     high=1.0,
                                     step=0.0001)

    # Set the normalize rewards flag.
    normalize_rewards_choices = [True, False]
    normalize_rewards = trial.suggest_categorical(name='normalize_rewards',
                                                  choices=normalize_rewards_choices)

    # Set the hyperparameters directly in `args`.
    args.lr = round(lr, 8)
    args.linear_lr_decay = linear_lr_decay
    args.gamma = round(gamma, 4)
    args.num_samples = num_samples
    args.mini_batch_size = mini_batch_size
    args.epochs = epochs
    args.epsilon = round(epsilon, 4)
    args.vf_loss_coef = round(vf_loss_coef, 4)
    args.policy_entropy_coef = round(policy_entropy_coef, 7)
    args.clipped_value_fn = clipped_value_fn
    args.max_grad_norm = max_grad_norm
    args.use_gae = use_gae
    args.gae_lambda = round(gae_lambda, 4)
    args.normalize_rewards = normalize_rewards
    args.wandb = True

    # Define the seeds for the experiment.
    seeds = [0, 1, 2, 3, 4]

    cumulative_returns = []
    for seed in seeds:

        # Set the random seed for the experiment.
        args.seed = seed

        # Create a new controller object.
        controller = NormalController()

        # Run training.
        controller.run()

        # Compute the cumulative return.
        seed_returns = [x[-2] for x in controller.eval_data]
        seed_cumulative_return = np.sum(seed_returns)

        # Append the cumulative return to the list.
        cumulative_returns.append(seed_cumulative_return)

    # Compute the average cumulative return across the seeds.
    average_cumulative_returns = np.average(cumulative_returns)

    return average_cumulative_returns


def main():

    if args.optuna:

        env = args.n_env_name.split('-')[0].lower()
        version = args.n_env_name.split('-')[1].lower()
        prefix = f'{env}{version}'

        optuna_folder = f'{os.getenv("HOME")}/Documents/openai/optuna'
        os.makedirs(optuna_folder, exist_ok=True)

        study_name = f'{prefix}_ppo_optuna_study'
        storage = f'sqlite:///{optuna_folder}/{prefix}_ppo_optuna_study.db'
        sampler = optuna.samplers.TPESampler(n_startup_trials=50)
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    direction='maximize',
                                    load_if_exists=True,
                                    sampler=sampler)

        def print_trial_count(study, trial):
            print(f'Trial {trial.number} completed. Total trials so far: {len(study.trials)}\n')

        study.optimize(
            objective,
            n_trials=1,
            callbacks=[print_trial_count]
        )

        with open(f'{optuna_folder}/{prefix}_ppo_optuna.txt', 'w') as f:
            print(f'Best hyperparameters found:', file=f)
            for key, value in study.best_params.items():
                print(f'{key}: {value}', file=f)
            print('\n', file=f)
            print(f'Best average return:\n{study.best_value}', file=f)

    else:

        nc = NormalController()

        try:

            nc.run()

        except KeyboardInterrupt as e:

            print('keyboard interrupt')


if __name__ == '__main__':

    main()
