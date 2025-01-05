import argparse
import csv
import itertools
import matplotlib.pyplot as plt
import optuna
import os

# The following three lines must come before numpy import.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
from controllers.sac.sac_agent import SAC
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs  # do not delete; required for custom gym environments

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Arguments")

parser.add_argument("-e", "--n_env_name", default="Ant-v5",
                    help="name of normal (non-malfunctioning) MuJoCo Gym environment (default: Ant-v5)")
parser.add_argument("-t", "--n_time_steps", type=int, default=3000000, metavar="N",
                    help="number of time steps in normal (non-malfunctioning) MuJoCo Gym environment (default: 3000000)")

parser.add_argument("--gamma", type=float, default=0.99, metavar="G",
                    help="discount factor (default: 0.99)")
parser.add_argument("--tau", type=float, default=0.01, metavar="G",
                    help="target smoothing coefficient (default: 0.01)")
parser.add_argument("--alpha", type=float, default=0.2, metavar="G",
                    help="temperature parameter that determines the relative importance of the entropy term against the reward (default: 0.2)")
parser.add_argument("--lr", type=float, default=0.0003, metavar="G",
                    help="learning rate (default: 0.0003)")

parser.add_argument("--hidden_dim", type=int, default=256, metavar="N",
                    help="hidden dimension (default: 256)")

parser.add_argument("-rbs", "--replay_buffer_size", type=int, default=1000000, metavar="N",
                    help="size of replay buffer (default: 1000000)")
parser.add_argument("--batch_size", type=int, default=256, metavar="N",
                    help="number of samples per batch (default: 256)")

parser.add_argument("-nr", "--normalize_rewards", default=False, action="store_true",
                    help="if true, normalize rewards in memory (default: False)")

parser.add_argument("--model_updates_per_step", type=int, default=1, metavar="N",
                    help="number of model updates per simulator step (default: 1)")
parser.add_argument("--target_update_interval", type=int, default=1, metavar="N",
                    help="number of target value network updates per number of gradient steps (network updates) (default: 1)")

parser.add_argument("-a", "--automatic_entropy_tuning", default=False, action="store_true",
                    help="if true, automatically tune the temperature (default: False)")

parser.add_argument("-tef", "--time_step_eval_frequency", type=int, default=15000, metavar="N",
                    help="frequency of policy evaluation during learning (default: 7500)")
parser.add_argument("-ee", "--eval_episodes", type=int, default=10, metavar="N",
                    help="number of episodes in policy evaluation roll-out (default: 10)")

parser.add_argument("-c", "--cuda", default=False, action="store_true",
                    help="if true, run on GPU (default: False)")

parser.add_argument("-s", "--seed", type=int, default=0, metavar="N",
                    help="random seed (default: 0)")

parser.add_argument("-d", "--delete", default=False, action="store_true",
                    help="if true, delete previously saved data and restart training (default: False)")

parser.add_argument("-wb", "--wandb", default=False, action="store_true",
                    help="if true, log to weights & biases (default: False)")

parser.add_argument("-o", "--optuna", default=False, action="store_true",
                    help="if true, run a parameter search with optuna (default: False)")

args = parser.parse_args()


class NormalController:
    """
    Controller for learning in the normal environment.

    The experiment program directs the experiment's execution, including the sequence of agent-environment interactions
    and agent performance evaluation.  -Brian Tanner & Adam White
    """

    LINE = "--------------------------------------------------------------------------------"

    def __init__(self):

        # experiment runtime information

        self.start = time.time()

        # experiment parameters

        self.parameters = None

        self.parameters = {
            "n_env_name": args.n_env_name,
            "n_time_steps": args.n_time_steps,
            "gamma": args.gamma,
            "tau": args.tau,
            "alpha": args.alpha,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "replay_buffer_size": args.replay_buffer_size,
            "batch_size": args.batch_size,
            "normalize_rewards": args.normalize_rewards,
            "model_updates_per_step": args.model_updates_per_step,
            "target_update_interval": args.target_update_interval,
            "automatic_entropy_tuning": args.automatic_entropy_tuning,
            "time_step_eval_frequency": args.time_step_eval_frequency,
            "eval_episodes": args.eval_episodes,
            "cuda": args.cuda,
            "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu",
            "seed": args.seed,
            "wandb": args.wandb,
            "optuna": args.optuna
        }

        # W&B initialization

        if self.parameters["wandb"]:

            wandb.init(
                project="sac_antv5",
                config=self.parameters,
                dir=f'{os.getenv("HOME")}/Documents/openai'
            )

            wandb.define_metric(
                name="Key Metrics/*",
                step_metric="Time Steps"
            )

            wandb.define_metric(
                name="Loss Metrics/*",
                step_metric="Number of Updates"
            )

        # experiment data directory

        suffix = (
            f'{self.parameters["n_env_name"]}:{self.parameters["n_time_steps"]}'
            f'_g:{self.parameters["gamma"]}'
            f'_t:{self.parameters["tau"]}'
            f'_a:{self.parameters["alpha"]}'
            f'_lr:{self.parameters["lr"]}'
            f'_hd:{self.parameters["hidden_dim"]}'
            f'_rbs:{self.parameters["replay_buffer_size"]}'
            f'_bs:{self.parameters["batch_size"]}'
            f'_nr:{self.parameters["normalize_rewards"]}'
            f'_mups:{self.parameters["model_updates_per_step"]}'
            f'_tui:{self.parameters["target_update_interval"]}'
            f'_a:{self.parameters["automatic_entropy_tuning"]}'
            f'_tef:{self.parameters["time_step_eval_frequency"]}'
            f'_ee:{self.parameters["eval_episodes"]}'
            f'_d:{self.parameters["device"]}'
            f'{"_wb" if self.parameters["wandb"] else ""}'
            f'{"_o" if self.parameters["optuna"] else ""}'
        )

        self.experiment = f'SAC_{suffix}'

        self.data_dir = f'{os.getenv("HOME")}/Documents/openai/data/{self.experiment}/seed{self.parameters["seed"]}'

        # are we restarting training?  do the data files for the
        # selected seed already exist?

        if path.exists(self.data_dir):

            print(self.LINE)
            print(self.LINE)

            if args.delete:
                # yes; argument flag present to indicate data deletion
                print(colored("argument indicates DATA DELETION", "red"))
                print(colored("deleting data...", "red"))
                rmtree(self.data_dir, ignore_errors=True)
                print(colored("data deletion complete", "red"))
            else:
                # yes; argument flag not present; get confirmation of data deletion from user input
                print(colored("You are about to delete saved data and restart training.", "red"))
                s = input(colored("Are you sure you want to continue?  Hit 'y' then 'Enter' to continue.\n", "red"))
                if s == "y":
                    # delete old data; rewrite new data to same location
                    print(colored("user input indicates DATA DELETION", "red"))
                    print(colored("deleting data...", "red"))
                    rmtree(self.data_dir, ignore_errors=True)
                    print(colored("data deletion complete", "red"))
                else:
                    # do not delete old data; system exit
                    print(colored("user input indicates NO DATA DELETION", "red"))
                    print(self.LINE)
                    sys.exit("\nexiting...")

        # data

        num_rows = int(self.parameters["n_time_steps"] / self.parameters["time_step_eval_frequency"]) + 1  # add 1 for evaluation before any learning (0th entry)
        num_columns = 5
        self.eval_data = np.zeros((num_rows, num_columns))

        # num_rows = self.parameters["n_time_steps"]  # larger than needed; will remove extra entries later
        # num_columns = 3
        # self.train_data = np.zeros((num_rows, num_columns))

        num_rows = (self.parameters["n_time_steps"] - self.parameters["batch_size"]) * self.parameters["model_updates_per_step"]
        num_columns = 7
        self.loss_data = np.zeros((num_rows, num_columns))

        # seeds

        random.seed(self.parameters["seed"])
        np.random.seed(self.parameters["seed"])
        torch.manual_seed(self.parameters["seed"])
        if self.parameters["device"] == "cuda":
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
        self.agent = SAC(self.env.env_observation_dim(),
                         self.env.env_action_dim(),
                         self.parameters["gamma"],
                         self.parameters["tau"],
                         self.parameters["alpha"],
                         self.parameters["lr"],
                         self.parameters["hidden_dim"],
                         self.parameters["replay_buffer_size"],
                         self.parameters["batch_size"],
                         self.parameters["normalize_rewards"],
                         self.parameters["model_updates_per_step"],
                         self.parameters["target_update_interval"],
                         self.parameters["automatic_entropy_tuning"],
                         self.parameters["device"],
                         self.parameters["wandb"],
                         self.loss_data)

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = {"num_episodes": 0,
                               "num_steps": 0,
                               "total_reward": 0}

        # print summary info

        # is GPU being used?

        print(self.LINE)

        if self.parameters["device"] == "cuda":
            print("NOTE: GPU is being used for this experiment.")
        else:
            print("NOTE: GPU is not being used for this experiment.  CPU only.")

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
                return colored(self.parameters[argument], "red")
            else:
                return self.parameters[argument]

        print("normal environment name:", highlight_non_default_values("n_env_name"))
        print("normal time steps:", highlight_non_default_values("n_time_steps"))
        print("gamma:", highlight_non_default_values("gamma"))
        print("tau:", highlight_non_default_values("tau"))
        print("alpha:", highlight_non_default_values("alpha"))
        print("learning rate:", highlight_non_default_values("lr"))
        print("hidden dimension:", highlight_non_default_values("hidden_dim"))
        print("replay buffer size:", highlight_non_default_values("replay_buffer_size"))
        print("batch size:", highlight_non_default_values("batch_size"))
        print("normalize rewards:", highlight_non_default_values("normalize_rewards"))
        print("model updates per step:", highlight_non_default_values("model_updates_per_step"))
        print("target update interval:", highlight_non_default_values("target_update_interval"))
        print("automatic entropy tuning:", highlight_non_default_values("automatic_entropy_tuning"))
        print("time step evaluation frequency:", highlight_non_default_values("time_step_eval_frequency"))
        print("evaluation episodes:", highlight_non_default_values("eval_episodes"))
        if self.parameters["device"] == "cuda":
            print("device:", self.parameters["device"])
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print("cuda visible device(s):", colored(os.environ["CUDA_VISIBLE_DEVICES"], "red"))
            else:
                print(colored("cuda visible device(s): N/A", "red"))
        else:
            print("device:", colored(self.parameters["device"], "red"))
        print("seed:", colored(self.parameters["seed"], "red"))
        print("wandb:", highlight_non_default_values("wandb"))
        print("optuna:", highlight_non_default_values("optuna"))

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

            # index = self.rlg.num_episodes() - 1
            # self.train_data[index] = [self.rlg.num_episodes(), self.rlg.num_steps(), self.rlg.episode_reward()]

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
        Send completion email.
        Save file with run information.
        Close wandb.
        """

        self.rlg.rl_env_message("close")

        run_time = str(timedelta(seconds=time.time() - self.start))[:-7]

        print("time to complete one run:", run_time, "h:m:s")
        print(self.LINE)

        text_file = open(self.data_dir + "/run_summary.txt", "w")
        text_file.write(date.today().strftime("%m/%d/%y"))
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

            # save torch RNG state(s)
            # this is needed because the SAC network sample() method
            # samples from a normal distribution during evaluation,
            # despite only needing a deterministic (mean) sample.
            rng_state_cpu = torch.get_rng_state()
            if torch.cuda.is_available():
                rng_state_cuda = torch.cuda.get_rng_state()
            else:
                rng_state_cuda = None

            eval_agent = copy(self.agent)

            eval_rlg = RLGlue(self.eval_env, eval_agent)

            eval_rlg.rl_agent_message("mode, eval")

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

            if num_time_steps == 0:
                num_updates = 0
            else:
                num_updates = (num_time_steps - self.parameters["batch_size"]) * self.parameters["model_updates_per_step"]

            num_samples = num_updates * self.parameters["batch_size"]

            real_time = int(time.time() - self.start)

            index = num_time_steps // self.parameters["time_step_eval_frequency"]
            self.eval_data[index] = [num_time_steps,
                                     num_updates,
                                     num_samples,
                                     average_return,
                                     real_time]

            cumulative_average_return = self.eval_data[:, -2].sum()

            if self.parameters["wandb"]:
                data = {
                    "Key Metrics/Average Return": average_return,
                    "Key Metrics/Cumulative Average Return": cumulative_average_return,
                    "Key Metrics/Real Time": real_time,
                    "Time Steps": num_time_steps
                }
                wandb.log(data=data)

            print(f'evaluation at {num_time_steps} time steps: {average_return}')

            run_time = str(timedelta(seconds=time.time() - self.start))[:-7]
            print("runtime:", run_time, "h:m:s")
            print(self.LINE)

            # reload the torch RNG state(s)
            torch.set_rng_state(rng_state_cpu)
            if rng_state_cuda is not None:
                torch.cuda.set_rng_state(rng_state_cuda)

    def plot(self):
        """
        Plot experiment data.

        File format: .jpg
        """

        print("plotting...")

        csv_foldername = self.data_dir + "/csv"
        os.makedirs(csv_foldername, exist_ok=True)

        jpg_foldername = self.data_dir + "/jpg"
        os.makedirs(jpg_foldername, exist_ok=True)

        df = pd.read_csv(csv_foldername + "/eval_data.csv")

        # evaluation: average_return vs num_time_steps
        df.plot(x="num_time_steps", y="average_return", color="blue", legend=False)
        plt.xlabel("time_steps")
        plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Policy Evaluation")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/evaluation_time_steps.jpg")
        plt.close()

        # evaluation: average_return vs num_updates
        df.plot(x="num_updates", y="average_return", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Policy Evaluation")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/evaluation_updates.jpg")
        plt.close()

        # evaluation: average_return vs num_samples
        df.plot(x="num_samples", y="average_return", color="blue", legend=False)
        plt.xlabel("samples")
        plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Policy Evaluation")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/evaluation_samples.jpg")
        plt.close()

        df = pd.read_csv(csv_foldername + "/loss_data.csv")

        # training: q_value_loss_1 vs num_updates
        df.plot(x="num_updates", y="q_value_loss_1", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("Q Value Loss 1")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/q_value_loss_1.jpg")
        plt.close()

        # training: q_value_loss_2 vs num_updates
        df.plot(x="num_updates", y="q_value_loss_2", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("Q Value Loss 2")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/q_value_loss_2.jpg")
        plt.close()

        # training: policy_loss vs num_updates
        df.plot(x="num_updates", y="policy_loss", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("Policy Loss")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/policy_loss.jpg")
        plt.close()

        # training: alpha_loss vs num_updates
        df.plot(x="num_updates", y="alpha_loss", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("Alpha Loss")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/alpha_loss.jpg")
        plt.close()

        # training: alpha_value vs num_updates
        df.plot(x="num_updates", y="alpha_value", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("alpha", rotation="horizontal", labelpad=30)
        plt.title("Alpha Value")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/alpha_value.jpg")
        plt.close()

        # training: entropy vs num_updates
        df.plot(x="num_updates", y="entropy", color="blue", legend=False)
        plt.xlabel("updates")
        plt.ylabel("entropy", rotation="horizontal", labelpad=30)
        plt.title("Entropy")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/entropy_updates.jpg")
        plt.close()

        print("plotting complete")

        print(self.LINE)

    def save(self):
        """
        Save experiment data before policy evaluation begins.

        Save experiment parameters.
        Save experiment data: train data and loss data.
        Save seed state: random, torch, and numpy seed states.
        Save rlg statistics: num_episodes, num_steps, and total_reward.
        Save environment data: OpenAI seed states.
        Save agent data: models, number of updates of models, and replay buffer.
        """

        print("saving...")

        self.save_seed_state()

        self.save_parameters()

        self.save_data()

        # save rlg data
        self.save_rlg_statistics()

        # save environment data
        self.rlg.rl_env_message(f'save, {self.data_dir}')

        # save agent data
        self.rlg.rl_agent_message(f'save, {self.data_dir}, {self.rlg.num_steps()}')

        print("saving complete")

        print(self.LINE)

    def save_data(self):
        """
        Save experiment data.

        File format: .csv
        """

        csv_foldername = self.data_dir + "/csv"
        os.makedirs(csv_foldername, exist_ok=True)

        eval_data_df = pd.DataFrame({"num_time_steps": self.eval_data[:, 0],
                                     "num_updates": self.eval_data[:, 1],
                                     "num_samples": self.eval_data[:, 2],
                                     "average_return": self.eval_data[:, 3],
                                     "real_time": self.eval_data[:, 4]})
        eval_data_df.to_csv(csv_foldername + "/eval_data.csv", float_format="%f")

        loss_data_df = pd.DataFrame({"num_updates": self.loss_data[:, 0],
                                     "q_value_loss_1": self.loss_data[:, 1],
                                     "q_value_loss_2": self.loss_data[:, 2],
                                     "policy_loss": self.loss_data[:, 3],
                                     "alpha_loss": self.loss_data[:, 4],
                                     "alpha_value": self.loss_data[:, 5],
                                     "entropy": self.loss_data[:, 6]})
        loss_data_df.to_csv(csv_foldername + "/loss_data.csv", float_format="%f")

    def save_parameters(self):
        """
        Save experiment parameters.

        File format: .csv and .pickle
        """

        csv_foldername = self.data_dir + "/csv"
        os.makedirs(csv_foldername, exist_ok=True)

        csv_filename = csv_foldername + "/parameters.csv"

        f = open(csv_filename, "w")
        writer = csv.writer(f)
        for key, val in self.parameters.items():
            writer.writerow([key, val])
        f.close()

        pickle_foldername = self.data_dir + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        pickle_filename = pickle_foldername + "/parameters.pickle"

        with open(pickle_filename, "wb") as f:
            pickle.dump(self.parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_rlg_statistics(self):
        """
        Save rlg statistics: num_episodes, num_steps, and total_reward.

        File format: .pickle
        """

        self.rlg_statistics["num_episodes"] = self.rlg.num_episodes()
        self.rlg_statistics["num_steps"] = self.rlg.num_steps()
        self.rlg_statistics["total_reward"] = self.rlg.total_reward()

        pickle_foldername = self.data_dir + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        with open(pickle_foldername + "/rlg_statistics.pickle", "wb") as f:
            pickle.dump(self.rlg_statistics, f)

    def save_seed_state(self):
        """
        Save state of the random number generators used by random, numpy, and pytorch.

        File format: .pickle (random and numpy) and .pt (pytorch)
        """

        pickle_foldername = self.data_dir + "/pickle"
        os.makedirs(pickle_foldername, exist_ok=True)

        random_random_state = random.getstate()
        numpy_random_state = np.random.get_state()

        with open(pickle_foldername + "/random_random_state.pickle", "wb") as f:
            pickle.dump(random_random_state, f)

        with open(pickle_foldername + "/numpy_random_state.pickle", "wb") as f:
            pickle.dump(numpy_random_state, f)

        pt_foldername = self.data_dir + "/pt"
        os.makedirs(pt_foldername, exist_ok=True)

        torch_random_state = torch.get_rng_state()
        torch.save(torch_random_state, pt_foldername + "/torch_random_state.pt")

        if self.parameters["device"] == "cuda":
            torch_cuda_random_state = torch.cuda.get_rng_state()
            torch.save(torch_cuda_random_state, pt_foldername + "/torch_cuda_random_state.pt")


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning of the SAC
    agent.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object that provides
        parameter suggestions.

    Returns:
        float: The average return after training for the specified
        number of time steps.
    """

    # Set gamma.
    gamma = trial.suggest_float(name="gamma",
                                low=0.8,
                                high=0.9999,
                                step=0.0001)

    # Set tau.
    tau = trial.suggest_float(name="tau",
                              low=0.001,
                              high=0.1,
                              step=0.0000001)

    # Set alpha.
    alpha = trial.suggest_float(name="alpha",
                                low=0.0001,
                                high=0.2,
                                step=0.0000001)

    # Set the learning rate.
    lr = trial.suggest_float(name="lr",
                             low=0.00001,
                             high=0.001,
                             step=0.00000001)

    # Set the replay buffer size.
    replay_buffer_size_choices = [100000, 250000, 500000, 750000, 1000000]
    replay_buffer_size = trial.suggest_categorical(
        name="replay_buffer_size",
        choices=replay_buffer_size_choices
    )

    # Set the batch size.
    batch_size_choices = [64, 128, 256, 512]
    batch_size = trial.suggest_categorical(
        name="batch_size",
        choices=batch_size_choices
    )

    # Set the normalize rewards flag.
    normalize_rewards_choices = [True, False]
    normalize_rewards = trial.suggest_categorical(
        name="normalize_rewards",
        choices=normalize_rewards_choices
    )

    # # Set the model updates per step.
    # model_updates_per_step_choices = [1, 2, 3, 4, 5]
    # model_updates_per_step = trial.suggest_categorical(
    #     name="model_updates_per_step",
    #     choices=model_updates_per_step_choices
    # )

    # Set the target update interval.
    target_update_interval_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_update_interval = trial.suggest_categorical(
        name="time_step_evaluation_frequency",
        choices=target_update_interval_choices
    )

    # Set the automatic entropy tuning flag.
    automatic_entropy_tuning_choices = [True, False]
    automatic_entropy_tuning = trial.suggest_categorical(
        name="automatic_entropy_tuning",
        choices=automatic_entropy_tuning_choices
    )

    # Set the hyperparameters directly in `args`.
    args.gamma = round(gamma, 4)
    args.tau = round(tau, 7)
    args.alpha = round(alpha, 7)
    args.lr = round(lr, 8)
    args.replay_buffer_size = replay_buffer_size
    args.batch_size = batch_size
    args.normalize_rewards = normalize_rewards
    # args.model_updates_per_step = model_updates_per_step
    args.target_update_interval = target_update_interval
    args.automatic_entropy_tuning = automatic_entropy_tuning
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

        optuna_folder = f'{os.getenv("HOME")}/Documents/openai/optuna'
        os.makedirs(optuna_folder, exist_ok=True)

        study_name = 'sac_optuna_study'
        storage = f'sqlite:///{optuna_folder}/sac_optuna_study.db'
        sampler = optuna.samplers.TPESampler(n_startup_trials=50)
        study = optuna.create_study(study_name=study_name,
                                    storage=storage,
                                    direction="maximize",
                                    load_if_exists=True,
                                    sampler=sampler)

        def print_trial_count(study, trial):
            print(f'Trial {trial.number} completed. Total trials so far: {len(study.trials)}\n')

        study.enqueue_trial(
            {"gamma": 0.00058328,
             "tau": True,
             "alpha": 0.9785,
             "lr": 2048,
             "replay_buffer_size": 512,
             "batch_size": 5,
             "target_update_interval": 0.4932,
             "automatic_entropy_tuning": 0.0055157},
        )

        study.optimize(
            objective,
            n_trials=1,
            callbacks=[print_trial_count]
        )

        with open(f'{optuna_folder}/sac_optuna.txt', "w") as f:
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

            print("keyboard interrupt")


if __name__ == "__main__":

    main()
