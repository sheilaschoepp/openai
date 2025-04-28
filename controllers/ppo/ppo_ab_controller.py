import argparse
import csv
import itertools
import matplotlib.pyplot as plt
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

parser.add_argument('-e', '--ab_env_name', default='Ant-v5',
                    help='name of abnormal (malfunctioning) MuJoCo Gym environment (default: Ant-v5)')
parser.add_argument('-t', '--ab_time_steps', type=int, default=10000000, metavar='N',
                    help='number of time steps in abnormal (malfunctioning) MuJoCo Gym environment (default: 10000000)')

parser.add_argument("-cm", "--clear_memory", default=False, action="store_true",
                    help="if true, clear the memory (default: False)")
parser.add_argument("-rn", "--reinitialize_networks", default=False, action="store_true",
                    help="if true, randomly reinitialize the networks (default: False)")

parser.add_argument('-c', '--cuda', default=False, action='store_true',
                    help='if true, run on GPU (default: False)')

parser.add_argument('-f', '--file',
                    help='absolute path of the seedX folder containing data from normal MuJoCo environment')

parser.add_argument('-d', '--delete', default=False, action='store_true',
                    help='if true, delete previously saved data and restart training (default: False)')

parser.add_argument('-wb', '--wandb', default=False, action='store_true',
                    help='if true, log to weights & biases (default: False)')

args = parser.parse_args()


class AbnormalController:
    """
    Controller for learning in the abnormal environment.

    The experiment program directs the experiment's execution, including the sequence of agent-environment interactions
    and agent performance evaluation.  -Brian Tanner & Adam White
    """

    LINE = '--------------------------------------------------------------------------------'

    def __init__(self):

        # experiment runtime information

        self.start = time.time()

        # experiment parameters

        self.parameters = None

        self.load_data_dir = args.file

        self.load_parameters()
        self.parameters["ab_env_name"] = args.ab_env_name  # addition
        self.parameters["ab_time_steps"] = args.ab_time_steps  # addition
        self.parameters["clear_memory"] = args.clear_memory  # addition
        self.parameters["reinitialize_networks"] = args.reinitialize_networks  # addition
        self.parameters["cuda"] = args.cuda  # update
        self.parameters["file"] = args.file  # addition
        self.parameters["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"  # update
        self.parameters["wandb"] = args.wandb # update

        # W&B initialization

        if self.parameters["wandb"]:

            env = self.parameters["ab_env_name"].split('-')[0].lower()
            version = self.parameters["ab_env_name"].split('-')[1].lower()
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
            f'{self.parameters["ab_env_name"]}:{self.parameters["ab_time_steps"]}'
            f'_{self.parameters["n_env_name"]}:{self.parameters["n_time_steps"]}'
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
            f'_cm:{self.parameters["clear_memory"]}'
            f'_rn:{self.parameters["reinitialize_networks"]}'
            f'_d:{self.parameters["device"]}'
            f'{"_wb" if self.parameters["wandb"] else ""}'
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

        # data is loaded in call to self.load(), after rl problem is initialized
        self.eval_data = None
        # self.train_data = None
        self.loss_data = None

        # seeds

        # seeds loaded in call to self.load(), after rl problem is initialized
        # Note: we load the seeds after all rl problem elements are created because the creation of the agent
        # network(s) uses xavier initialization, thereby altering the torch seed state

        # env is seeded in Environment env_init() method

        # rl problem

        # environment used for training
        self.env = Environment(self.parameters["ab_env_name"],
                               self.parameters["seed"])

        # environment used for evaluation
        self.eval_env = Environment(self.parameters["ab_env_name"],
                                    self.parameters["seed"])

        # agent
        self.agent = PPO(self.env.env_observation_dim(),
                         self.env.env_action_dim(),
                         self.parameters["hidden_dim"],
                         self.parameters["log_std"],
                         self.parameters["lr"],
                         self.parameters["linear_lr_decay"],
                         self.parameters["gamma"],
                         self.parameters["ab_time_steps"],
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
        self.rlg_statistics = None

        # resume experiment - load data, seed state, env, agent, and rlg
        self.load()

        # clear the memory if indicated by argument
        if self.parameters["clear_memory"]:
            self.rlg.rl_agent_message("clear_memory")

        # reinitialize the networks if indicated by argument
        if self.parameters["reinitialize_networks"]:
            self.rlg.rl_agent_message("reinitialize_networks")

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

        print(f'abnormal environment name: {highlight_non_default_values("ab_env_name")}')
        print(f'abnormal time steps: {highlight_non_default_values("ab_time_steps")}')
        print(f'normal environment name: {self.parameters["n_env_name"]}')
        print(f'normal time steps: {self.parameters["n_time_steps"]}')
        print(f'learning rate: {self.parameters["lr"]}')
        print("linear lr decay:", self.parameters["linear_lr_decay"])
        print(f'gamma: {self.parameters["gamma"]}')
        print("number of samples:", self.parameters["num_samples"])
        print("mini-batch size:", self.parameters["mini_batch_size"])
        print("epochs:", self.parameters["epochs"])
        print("epsilon:", self.parameters["epsilon"])
        print("value function loss coefficient:", self.parameters["vf_loss_coef"])
        print("policy entropy coefficient:", self.parameters["policy_entropy_coef"])
        print("clipped value function:", self.parameters["clipped_value_fn"])
        print("max norm of gradients:", self.parameters["max_grad_norm"])
        print("use generalized advantage estimation:", self.parameters["use_gae"])
        print("gae smoothing coefficient (lambda):", self.parameters["gae_lambda"])
        print(f'hidden dimension: {self.parameters["hidden_dim"]}')
        print("log_std:", self.parameters["log_std"])
        print("time step evaluation frequency:", self.parameters["time_step_eval_frequency"])
        print(f'evaluation episodes: {self.parameters["eval_episodes"]}')
        print("clear memory:", highlight_non_default_values("clear_memory"))
        print(f'reinitialize networks: {highlight_non_default_values("reinitialize_networks")}')

        if self.parameters['device'] == 'cuda':
            print(f'device: {self.parameters["device"]}')
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                print(f'cuda visible device(s): {colored(os.environ["CUDA_VISIBLE_DEVICES"], "red")}')
            else:
                print(colored('cuda visible device(s): N/A', 'red'))
        else:
            print(f'device: {colored(self.parameters["device"], "red")}')

        print(f'seed: {colored(self.parameters["seed"], "red")}')
        print(f'wandb: {highlight_non_default_values("wandb")}')

        print(self.LINE)
        print(self.LINE)

    def run(self):
        """
        Run Experiment.
        """

        self.rlg.rl_init(total_reward=self.rlg_statistics["total_reward"],
                         num_steps=self.rlg_statistics["num_steps"],
                         num_episodes=self.rlg_statistics["num_episodes"])

        # with the new gymnasium seeding method, we need to reload data
        # after the call to rl_init, which resets and seeds the environment
        self.rlg.rl_env_message(f'load, {self.load_data_dir}')

        # save the agent model and evaluate the model before any learning
        # self.rlg.rl_agent_message(f'save_model, {self.data_dir}, {self.parameters["n_time_steps"]}')  # not needed as we already have this model saved
        self.evaluate_model(self.rlg.num_steps())

        for _ in itertools.count(1):

            # episode time steps are limited to 1000 (set below)
            # this is used to ensure that once self.parameters["n_time_steps"] + self.parameters["ab_time_steps"] is reached, the experiment is terminated
            max_steps_this_episode = min(1000, self.parameters["n_time_steps"] + self.parameters["ab_time_steps"] - self.rlg.num_steps())

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
            if self.rlg.num_steps() == self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]:
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

            index = num_time_steps // self.parameters["time_step_eval_frequency"] + 1  # add 1 because we evaluate policy before learning
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

    def load(self):
        """
        Note: Experiment parameters already loaded.
        Load seed state: random, torch, and numpy seed states.
        Load experiment data.
        Load environment data: OpenAI seed states.
        Load agent data: models, number of updates of models, and memory.
        Load rlg statistics: num_episodes, num_steps, and total_reward.
        """

        self.load_seed_state()

        self.load_data()

        self.rlg.rl_env_message(f'load, {self.load_data_dir}')  # load environment data

        self.rlg.rl_agent_message(f'load, {self.load_data_dir}, {self.parameters["n_time_steps"]}')
        # consistency check: the following must be commented out for a
        # consistency check (see n_controller)
        self.rlg.rl_agent_message("reset_lr")
        self.agent.loss_data = self.loss_data

        self.load_rlg_statistics()  # load rlg data

    def load_data(self):
        """
        Load experiment data.

        File format: .csv
        """

        csv_foldername = f'{self.load_data_dir}/csv'

        num_rows = int(self.parameters["ab_time_steps"] / self.parameters["time_step_eval_frequency"]) + 1  # add 1 for evaluation before any learning (0th entry)
        num_columns = 3
        self.eval_data = pd.read_csv(f'{csv_foldername}/eval_data.csv').to_numpy().copy()[:, 1:]
        self.eval_data = np.append(self.eval_data, np.zeros((num_rows, num_columns)), axis=0)

        if self.parameters["clear_memory"]:
            num_rows = self.parameters["ab_time_steps"] // self.parameters["num_samples"]
        else:
            num_rows = ((self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]) // self.parameters["num_samples"]) - (self.parameters["n_time_steps"] // self.parameters["num_samples"])
        num_columns = 8
        self.loss_data = pd.read_csv(f'{csv_foldername}/loss_data.csv').to_numpy().copy()[:, 1:]
        self.loss_data = np.append(self.loss_data, np.zeros((num_rows, num_columns)), axis=0)

    def load_parameters(self):
        """
        Load normal experiment parameters.

        File format: .pickle
        """

        pickle_foldername = f'{self.load_data_dir}/pickle'

        with open(f'{pickle_foldername}/parameters.pickle', 'rb') as f:
            self.parameters = pickle.load(f)

    def load_rlg_statistics(self):
        """
        Load rlg statistics: num_episodes, num_steps, and total_reward.

        File format: .pickle
        """

        pickle_foldername = f'{self.load_data_dir}/pickle'

        with open(f'{pickle_foldername}/rlg_statistics.pickle', 'rb') as f:
            self.rlg_statistics = pickle.load(f)

    def load_seed_state(self):
        """
        Load state of the random number generators used by random, numpy, and pytorch.

        File format: .pickle and .pt
        """

        pickle_foldername = f'{self.load_data_dir}/pickle'

        with open(f'{pickle_foldername}/random_random_state.pickle', 'rb') as f:
            random_random_state = pickle.load(f)

        with open(f'{pickle_foldername}/numpy_random_state.pickle', 'rb') as f:
            numpy_random_state = pickle.load(f)

        random.setstate(random_random_state)
        np.random.set_state(numpy_random_state)

        pt_filename = f'{self.load_data_dir}/pt'

        torch_random_state = torch.load(f'{pt_filename}/torch_random_state.pt')
        torch.set_rng_state(torch_random_state)

        if self.parameters["device"] == 'cuda':
            if os.path.exists(f'{pt_filename}/torch_cuda_random_state.pt'):
                torch_cuda_random_state = torch.load(f'{pt_filename}/torch_cuda_random_state.pt')
                torch.cuda.set_rng_state(torch_cuda_random_state)
            else:
                torch.cuda.manual_seed(self.parameters["seed"])

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
        plt.axvline(x=self.parameters["n_time_steps"], ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel('time_steps')
        plt.ylabel('average\nreturn', rotation='horizontal', labelpad=30)
        plt.title('Policy Evaluation')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/evaluation_time_steps.jpg')
        plt.close()

        df = pd.read_csv(f'{csv_foldername}/loss_data.csv')

        # training: clip_loss vs num_updates
        df.plot(x='num_updates', y='clip_loss', color='blue', legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel('updates')
        plt.ylabel('loss', rotation='horizontal', labelpad=30)
        plt.title('CLIP Loss')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/clip_loss_updates.jpg')
        plt.close()

        # training: vf_loss vs num_updates
        df.plot(x='num_updates', y='vf_loss', color='blue', legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel('updates')
        plt.ylabel('loss', rotation='horizontal', labelpad=30)
        plt.title('VF Loss')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/vf_loss_updates.jpg')
        plt.close()

        # training: entropy vs num_updates
        df.plot(x='num_updates', y='entropy', color='blue', legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel('updates')
        plt.ylabel('entropy', rotation='horizontal', labelpad=30)
        plt.title('Entropy')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/entropy_updates.jpg')
        plt.close()

        # training: clip_vf_s_loss vs num_updates
        df.plot(x='num_updates', y='clip_vf_s_loss', color='blue', legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel('updates')
        plt.ylabel('loss', rotation='horizontal', labelpad=30)
        plt.title('CLIP+VF+S Loss')
        pss.plot_settings()
        plt.savefig(f'{jpg_foldername}/clip_vf_s_loss_updates.jpg')
        plt.close()

        # training: clip_fraction vs num_updates
        df.plot(x='num_updates', y='clip_fraction', color='blue', legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
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


def main():

    ac = AbnormalController()

    try:

        ac.run()

    except KeyboardInterrupt as e:

        print('keyboard interrupt')


if __name__ == '__main__':

    main()
