import argparse
import csv
import itertools
import os
os.environ["MKL_NUM_THREADS"] = "1"   # must be before numpy import
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import random
import smtplib
import sys
import time
from copy import copy
from datetime import date, timedelta
from os import path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from termcolor import colored

import utils.plot_style_settings as pss
from controllers.sacv2.sacv2_agent import SACv2
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs  # do not delete; required for custom gym environments

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Arguments")

parser.add_argument("-e", "--n_env_name", default="Ant-v2",
                    help="name of normal (non-malfunctioning) MuJoCo Gym environment (default: Ant-v2)")
parser.add_argument("-t", "--n_time_steps", type=int, default=20000000, metavar="N",
                    help="number of time steps in normal (non-malfunctioning) MuJoCo Gym environment (default: 20000000)")

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

parser.add_argument("--model_updates_per_step", type=int, default=1, metavar="N",
                    help="number of model updates per simulator step (default: 1)")
parser.add_argument("--target_update_interval", type=int, default=1, metavar="N",
                    help="number of target value network updates per number of gradient steps (network updates) (default: 1)")

parser.add_argument("-tef", "--time_step_eval_frequency", type=int, default=100000, metavar="N",
                    help="frequency of policy evaluation during learning (default: 100000)")
parser.add_argument("-ee", "--eval_episodes", type=int, default=10, metavar="N",
                    help="number of episodes in policy evaluation roll-out (default: 10)")
parser.add_argument("-tmsf", "--time_step_model_save_frequency", type=int, default=200000, metavar="N",
                    help="frequency of saving models during learning (default: 200000)")

parser.add_argument("-a", "--automatic_entropy_tuning", default=False, action="store_true",
                    help="if true, automatically tune the temperature (default: False)")

parser.add_argument("-c", "--cuda", default=False, action="store_true",
                    help="if true, run on GPU (default: False)")

parser.add_argument("-s", "--seed", type=int, default=0, metavar="N",
                    help="random seed (default: 0)")

parser.add_argument("-d", "--delete", default=False, action="store_true",
                    help="if true, delete previously saved data and restart training (default: False)")

parser.add_argument("--resumable", default=False, action="store_true",
                    help="if true, make experiment resumable (i.e. save data at the end of the last possible episode)")

parser.add_argument("--resume", default=False, action="store_true",
                    help="if true, resume experiment starting with data from last possible episode")

parser.add_argument("-rf", "--resume_file", default="",
                    help="absolute path of the seedX folder containing data from previous checkpoint")

parser.add_argument("-tl", "--time_limit", type=float, default=100000000000.0, metavar="N",
                    help="run time limit for use on Compute Canada (units: days)")

parser.add_argument("-ps", "--param_search", default=False, action="store_true",
                    help="if true, run a parameter search")

parser.add_argument("-pss", "--param_search_seed", type=int, default=0, metavar="N",
                    help="random seed for parameter search (default: 0)")

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

        # machines

        self.hostname = os.uname()[1]
        self.localhosts = ["melco", "Legion", "amii", "mehran"]
        self.computecanada = not any(host in self.hostname for host in self.localhosts)

        # experiment parameters

        self.parameters = None

        if not args.resume:

            self.parameters = {"n_env_name": args.n_env_name,
                               "n_time_steps": args.n_time_steps,
                               "gamma": args.gamma,
                               "tau": args.tau,
                               "alpha": args.alpha,
                               "lr": args.lr,
                               "hidden_dim": args.hidden_dim,
                               "replay_buffer_size": args.replay_buffer_size,
                               "batch_size": args.batch_size,
                               "model_updates_per_step": args.model_updates_per_step,
                               "target_update_interval": args.target_update_interval,
                               "time_step_eval_frequency": args.time_step_eval_frequency,
                               "eval_episodes": args.eval_episodes,
                               "time_step_model_save_frequency": args.time_step_model_save_frequency,
                               "automatic_entropy_tuning": args.automatic_entropy_tuning,
                               "cuda": args.cuda,
                               "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu",
                               "seed": args.seed,
                               "resumable": args.resumable,
                               "resume": args.resume,
                               "resume_file": args.resume_file,
                               "complete": False,
                               "param_search": args.param_search,
                               "param_search_seed": args.param_search_seed}

        else:

            self.load_data_dir = args.resume_file

            # load old parameters to make sure we are running the same experiment
            self.load_parameters()

            # overwrite parameters that need to be updated
            self.parameters["n_time_steps"] = args.n_time_steps  # overwrite
            self.parameters["cuda"] = args.cuda  # update
            self.parameters["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"  # update

            self.parameters["resumable"] = args.resumable  # update
            self.parameters["resume"] = args.resume  # update
            self.parameters["resume_file"] = args.resume_file  # update
            self.parameters["complete"] = False  # update

        # experiment data directory

        suffix = self.parameters["n_env_name"] + ":" + str(self.parameters["n_time_steps"]) \
                 + "_g:" + str(self.parameters["gamma"]) \
                 + "_t:" + str(self.parameters["tau"]) \
                 + "_a:" + str(self.parameters["alpha"]) \
                 + "_lr:" + str(self.parameters["lr"]) \
                 + "_hd:" + str(self.parameters["hidden_dim"]) \
                 + "_rbs:" + str(self.parameters["replay_buffer_size"]) \
                 + "_bs:" + str(self.parameters["batch_size"]) \
                 + "_mups:" + str(self.parameters["model_updates_per_step"]) \
                 + "_tui:" + str(self.parameters["target_update_interval"]) \
                 + "_tef:" + str(self.parameters["time_step_eval_frequency"]) \
                 + "_ee:" + str(self.parameters["eval_episodes"]) \
                 + "_tmsf:" + str(self.parameters["time_step_model_save_frequency"]) \
                 + "_a:" + str(self.parameters["automatic_entropy_tuning"]) \
                 + "_d:" + str(self.parameters["device"]) \
                 + (("_ps:" + str(self.parameters["param_search"])) if self.parameters["param_search"] else "") \
                 + (("_pss:" + str(self.parameters["param_search_seed"])) if self.parameters["param_search"] else "") \
                 + ("_r" if self.parameters["resumable"] else "") \
                 + ("_resumed" if self.parameters["resume"] else "")

        self.experiment = "SACv2_" + suffix

        if self.computecanada:
            # path for compute canada
            self.data_dir = os.getenv("HOME") + "/scratch/openai/data/" + self.experiment + "/seed" + str(self.parameters["seed"])
        else:
            # path for servers and local machines
            self.data_dir = os.getenv("HOME") + "/Documents/openai/data/" + self.experiment + "/seed" + str(self.parameters["seed"])

        # old data

        if not self.computecanada:
            # are we restarting training?  do the data files for the selected seed already exist?
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

        # new data

        num_rows = int(self.parameters["n_time_steps"] / self.parameters["time_step_eval_frequency"]) + 1  # add 1 for evaluation before any learning (0th entry)
        num_columns = 5
        self.eval_data = np.zeros((num_rows, num_columns))

        num_rows = self.parameters["n_time_steps"]  # larger than needed; will remove extra entries later
        num_columns = 3
        self.train_data = np.zeros((num_rows, num_columns))

        num_rows = (self.parameters["n_time_steps"] - self.parameters["batch_size"]) * self.parameters["model_updates_per_step"]
        num_columns = 6
        self.loss_data = np.zeros((num_rows, num_columns))

        # seeds

        random.seed(self.parameters["seed"])
        np.random.seed(self.parameters["seed"])
        torch.manual_seed(self.parameters["seed"])
        if self.parameters["device"] == "cuda":
            torch.cuda.manual_seed(self.parameters["seed"])

        # env is seeded in Environment __init__() method

        # rl problem

        # normal environment used for training
        self.env = Environment(self.parameters["n_env_name"],
                               self.parameters["seed"])

        # agent
        self.agent = SACv2(self.env.env_state_dim(),
                           self.env.env_action_dim(),
                           self.parameters["gamma"],
                           self.parameters["tau"],
                           self.parameters["alpha"],
                           self.parameters["lr"],
                           self.parameters["hidden_dim"],
                           self.parameters["replay_buffer_size"],
                           self.parameters["batch_size"],
                           self.parameters["model_updates_per_step"],
                           self.parameters["target_update_interval"],
                           self.parameters["automatic_entropy_tuning"],
                           self.parameters["device"],
                           self.loss_data)

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = {"num_episodes": 0, "num_steps": 0, "total_reward": 0}

        # resume experiment - load data, seed state, env, agent, and rlg

        if args.resume:

            self.load()

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

            @param argument: string
                the argument name

            @return: string
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
        print("model updates per step:", highlight_non_default_values("model_updates_per_step"))
        print("target updates interval:", highlight_non_default_values("target_update_interval"))
        print("time step evaluation frequency:", highlight_non_default_values("time_step_eval_frequency"))
        print("evaluation episodes:", highlight_non_default_values("eval_episodes"))
        print("time step model save frequency:", highlight_non_default_values("time_step_model_save_frequency"))
        print("automatic entropy tuning:", highlight_non_default_values("automatic_entropy_tuning"))
        if self.parameters["device"] == "cuda":
            print("device:", self.parameters["device"])
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print("cuda visible device(s):", colored(os.environ["CUDA_VISIBLE_DEVICES"], "red"))
            else:
                print(colored("cuda visible device(s): N/A", "red"))
        else:
            print("device:", colored(self.parameters["device"], "red"))
        print("seed:", colored(self.parameters["seed"], "red"))
        if self.parameters["param_search"]:
            print("param search:", colored(self.parameters["param_search"], "red"))
            print("param search seed:", colored(self.parameters["param_search_seed"], "red"))
        print("resumable:", highlight_non_default_values("resumable"))

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
        if not args.resume:
            self.rlg.rl_agent_message("save_model, {}, {}".format(self.data_dir, 0))
            self.evaluate_model(self.rlg.num_steps())

        for _ in itertools.count(1):

            # Compute Canada limits time usage; this prevents data loss
            run_time = (time.time() - self.start) / 3600  # units: hours
            allowed_time = (args.time_limit * 24) - 6 + (self.parameters["seed"] / 6)  # allow the last 6 hours to be used to save data for each seed (saving cannot happen at the same time or memory will run out)

            if run_time > allowed_time:
                print("allowed time of {} exceeded at a runtime of {}\nstopping experiment".format(str(timedelta(hours=allowed_time))[:-7], str(timedelta(hours=run_time))[:-7]))
                print(self.LINE)
                self.parameters["resumable"] = True
                self.parameters["completed_time_steps"] = self.rlg.num_steps()
                break

            # episode time steps are limited to 1000 (set below)
            # this is used to ensure that once self.parameters["n_time_steps"] is reached, the experiment is terminated
            max_steps_this_episode = min(1000, self.parameters["n_time_steps"] - self.rlg.num_steps())

            # if we want to make an experiment resumable, we must save after the last possible episode
            # since episodes are limited to be a maximum of 1000 time steps, we can save when the max_steps_this_episode is less than 1000
            if args.resumable and max_steps_this_episode < 1000:
                self.parameters["completed_time_steps"] = self.rlg.num_steps()
                break

            # run an episode
            self.rlg.rl_start()

            terminal = False

            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, _, terminal, _ = self.rlg.rl_step()

                # save the agent model each 'self.parameters["time_step_model_save_frequency"]' time steps
                # policy model will be used for videos demonstrating learning progress
                if self.rlg.num_steps() % self.parameters["time_step_model_save_frequency"] == 0:
                    self.rlg.rl_agent_message("save_model, {}, {}".format(self.data_dir, self.rlg.num_steps()))

                # evaluate the model every 'self.parameters["time_step_eval_frequency"]' time steps
                if self.rlg.num_steps() % self.parameters["time_step_eval_frequency"] == 0:
                    self.evaluate_model(self.rlg.num_steps())

            index = self.rlg.num_episodes() - 1
            self.train_data[index] = [self.rlg.num_episodes(), self.rlg.num_steps(), self.rlg.episode_reward()]

            # learning complete
            if self.rlg.num_steps() == self.parameters["n_time_steps"]:
                if self.parameters["resumable"]:
                    self.parameters["completed_time_steps"] = self.rlg.num_steps()
                else:
                    if "completed_time_steps" in self.parameters:
                        del self.parameters["completed_time_steps"]  # we no longer need this information
                    self.parameters["complete"] = True
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
        """

        self.rlg.rl_env_message("close")

        run_time = str(timedelta(seconds=time.time() - self.start))[:-7]
        print("time to complete one run:", run_time, "h:m:s")
        print(self.LINE)

        if not self.computecanada:
            self.send_email(run_time)

        text_file = open(self.data_dir + "/run_summary.txt", "w")
        text_file.write(date.today().strftime("%m/%d/%y"))
        text_file.write("\n\nExperiment {}/seed{} complete.\n\nTime to complete: {} h:m:s".format(self.experiment, self.parameters["seed"], run_time))
        text_file.close()

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

            agent_eval = copy(self.agent)
            env_eval = copy(self.env)
            env_eval.env_seed(self.parameters["seed"])

            rlg_eval = RLGlue(env_eval, agent_eval)

            rlg_eval.rl_agent_message("mode, eval")

            returns = []

            rlg_eval.rl_init()

            for e in range(self.parameters["eval_episodes"]):

                rlg_eval.rl_start()

                terminal = False

                max_steps_this_episode = 1000
                while not terminal and ((max_steps_this_episode <= 0) or (rlg_eval.num_ep_steps() < max_steps_this_episode)):
                    _, _, terminal, _ = rlg_eval.rl_step()

                returns.append(rlg_eval.episode_reward())

            average_return = np.average(returns)

            if num_time_steps == 0:
                num_updates = 0
            else:
                num_updates = (num_time_steps - self.parameters["batch_size"]) * self.parameters["model_updates_per_step"]
            num_samples = num_updates * self.parameters["batch_size"]

            real_time = int(time.time() - self.start)

            index = num_time_steps // self.parameters["time_step_eval_frequency"]
            self.eval_data[index] = [num_time_steps, num_updates, num_samples, average_return, real_time]

            print("evaluation at {} time steps: {}".format(num_time_steps, average_return))

            run_time = str(timedelta(seconds=time.time() - self.start))[:-7]
            print("runtime:", run_time, "h:m:s")
            print(self.LINE)

    def load(self):
        """
        Note: Experiment parameters already loaded.
        Load seed state: random, torch, and numpy seed states.
        Load experiment data.
        Load environment data: OpenAI seed states.
        Load agent data: models, number of updates of models, and replay buffer.
        Load rlg statistics: num_episodes, num_steps, and total_reward.
        """

        self.load_seed_state()

        self.load_data()

        self.rlg.rl_env_message("load, {}".format(self.load_data_dir))  # load environment data

        self.rlg.rl_agent_message("load, {}, {}".format(self.load_data_dir, self.parameters["completed_time_steps"]))  # load agent data
        self.agent.loss_data = self.loss_data

        self.load_rlg_statistics()  # load rlg data

    def load_data(self):
        """
        Load experiment data.

        File format: .csv
        """

        csv_foldername = self.load_data_dir + "/csv"

        self.eval_data = pd.read_csv(csv_foldername + "/eval_data.csv").to_numpy().copy()[:, 1:]
        num_rows = (self.parameters["n_time_steps"] // self.parameters["time_step_eval_frequency"]) + 1 - self.eval_data.shape[0]
        num_columns = self.eval_data.shape[1]
        if num_rows > 0:
            self.eval_data = np.append(self.eval_data, np.zeros((num_rows, num_columns)), axis=0)

        self.train_data = pd.read_csv(csv_foldername + "/train_data.csv").to_numpy().copy()[:, 1:]
        num_rows = self.parameters["n_time_steps"] - self.train_data.shape[0]  # always larger than needed; will remove extra entries later
        num_columns = self.train_data.shape[1]
        if num_rows > 0:
            self.train_data = np.append(self.train_data, np.zeros((num_rows, num_columns)), axis=0)

        self.loss_data = pd.read_csv(csv_foldername + "/loss_data.csv").to_numpy().copy()[:, 1:]
        num_rows = ((self.parameters["n_time_steps"] - self.parameters["batch_size"]) * self.parameters["model_updates_per_step"]) - self.loss_data.shape[0]
        num_columns = self.loss_data.shape[1]
        if num_rows > 0:
            self.loss_data = np.append(self.loss_data, np.zeros((num_rows, num_columns)), axis=0)

    def load_parameters(self):
        """
        Load normal experiment parameters.

        File format: .pickle
        """

        pickle_foldername = self.load_data_dir + "/pickle"

        with open(pickle_foldername + "/parameters.pickle", "rb") as f:
            self.parameters = pickle.load(f)

    def load_rlg_statistics(self):
        """
        Load rlg statistics: num_episodes, num_steps, and total_reward.

        File format: .pickle
        """

        pickle_foldername = self.load_data_dir + "/pickle"

        with open(pickle_foldername + "/rlg_statistics.pickle", "rb") as f:
            self.rlg_statistics = pickle.load(f)

    def load_seed_state(self):
        """
        Load state of the random number generators used by random, numpy, and pytorch.

        File format: .pickle and .pt
        """

        pickle_foldername = self.load_data_dir + "/pickle"

        with open(pickle_foldername + "/random_random_state.pickle", "rb") as f:
            random_random_state = pickle.load(f)

        with open(pickle_foldername + "/numpy_random_state.pickle", "rb") as f:
            numpy_random_state = pickle.load(f)

        random.setstate(random_random_state)
        np.random.set_state(numpy_random_state)

        pt_filename = self.load_data_dir + "/pt"

        torch_random_state = torch.load(pt_filename + "/torch_random_state.pt")
        torch.set_rng_state(torch_random_state)

        if self.parameters["device"] == "cuda":
            if os.path.exists(pt_filename + "/torch_cuda_random_state.pt"):
                torch_cuda_random_state = torch.load(pt_filename + "/torch_cuda_random_state.pt")
                torch.cuda.set_rng_state(torch_cuda_random_state)
            else:
                torch.cuda.manual_seed(self.parameters["seed"])

    def plot(self):
        """
        Plot experiment data.

        File format: .jpg
        """

        print("plotting...")

        csv_foldername = self.data_dir + "/csv"

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

        df = pd.read_csv(csv_foldername + "/train_data.csv")

        # training: episode_return vs num_episodes
        df.plot(x="num_episodes", y="episode_return", color="blue", legend=False)
        plt.xlabel("episodes")
        plt.ylabel("episode\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Training")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/train_episodes.jpg")
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

        self.save_rlg_statistics()  # save rlg data

        self.rlg.rl_env_message("save, {}".format(self.data_dir))  # save environment data

        self.rlg.rl_agent_message("save, {}, {}".format(self.data_dir, self.rlg.num_steps()))  # save agent data

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

        # remove zero entries
        index = None
        for i in range(self.train_data.shape[0]):
            if (self.train_data[i] == np.zeros(3)).all() and (self.train_data[i+1] == np.zeros(3)).all():
                index = i
                break
        self.train_data = self.train_data[:index]
        train_data_df = pd.DataFrame({"num_episodes": self.train_data[:, 0],
                                      "num_time_steps": self.train_data[:, 1],
                                      "episode_return": self.train_data[:, 2]})
        train_data_df.to_csv(csv_foldername + "/train_data.csv", float_format="%f")

        loss_data_df = pd.DataFrame({"num_updates": self.loss_data[:, 0],
                                     "q_value_loss_1": self.loss_data[:, 1],
                                     "q_value_loss_2": self.loss_data[:, 2],
                                     "policy_loss": self.loss_data[:, 3],
                                     "alpha_loss": self.loss_data[:, 4],
                                     "alpha_value": self.loss_data[:, 5]})
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

    def send_email(self, run_time):
        """
        Send email to indicate that the experiment is complete.

        @param run_time: string
            the time to complete a single run of the experiment (h:m:s)
        """

        gmail_email = "mynewbfnao@gmail.com"
        gmail_password = "k!1t8qL(YQO%labr}kS%"

        recipient = "sschoepp@ualberta.ca"

        from_ = gmail_email
        to = recipient if type(recipient) is list else [recipient]
        subject = "Experiment Complete"
        text = "Experiment {}/seed{} complete.\n\nTime to complete: \n{} h:m:s\n\nThis message is sent from Python.".format(self.experiment, self.parameters["seed"], run_time)

        message = """\From: %s\nTo: %s\nSubject: %s\n\n%s""" % (from_, ", ".join(to), subject, text)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(gmail_email, gmail_password)
        server.sendmail(from_, to, message)
        server.close()


def main():

    if args.param_search:
        param_search()

    nc = NormalController()

    try:

        nc.run()

    except KeyboardInterrupt as e:

        print("keyboard interrupt")


def param_search():
    """
    Conduct a random parameter search for SACv2 using parameter ranges.
    """

    np.random.seed(args.param_search_seed)

    args.gamma = round(np.random.uniform(0.8, 0.9997), 4)

    args.lr = round(np.random.uniform(0.000005, 0.006), 6)

    args.tau = round(np.random.uniform(0.0001, 0.1), 4)

    args.replay_buffer_size = int(np.random.choice([10000, 100000, 500000, 1000000]))

    args.batch_size = int(np.random.choice([16, 64, 256, 512]))


if __name__ == "__main__":

    main()
