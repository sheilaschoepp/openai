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
from controllers.ppov2.ppov2_agent import PPOv2
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs  # do not delete; required for custom gym environments

parser = argparse.ArgumentParser(description="PyTorch Proximal Policy Optimization Arguments")

parser.add_argument("-e", "--ab_env_name", default="Ant-v2",
                    help="name of abnormal (malfunctioning) MuJoCo Gym environment (default: Ant-v2)")
parser.add_argument("-t", "--ab_time_steps", type=int, default=400000000, metavar="N",
                    help="number of time steps in abnormal (malfunctioning) MuJoCo Gym environment (default: 400000000)")

parser.add_argument("-cm", "--clear_memory", default=False, action="store_true",
                    help="if true, clear the memory (default: False)")
parser.add_argument("-rn", "--reinitialize_networks", default=False, action="store_true",
                    help="if true, randomly reinitialize the networks (default: False)")

parser.add_argument("-c", "--cuda", default=False, action="store_true",
                    help="if true, run on GPU (default: False)")

parser.add_argument("-f", "--file",
                    help="absolute path of the seedX folder containing data from normal MuJoCo environment")

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

args = parser.parse_args()


class AbnormalController:
    """
    Controller for learning in the abnormal environment.

    The experiment program directs the experiment's execution, including the sequence of agent-environment interactions
    and agent performance evaluation.  -Brian Tanner & Adam White
    """

    LINE = "--------------------------------------------------------------------------------"

    def __init__(self):

        # experiment runtime information

        self.start = time.time()

        # hostname

        self.hostname = os.uname()[1]
        self.localhosts = ["melco", "Legion", "amii", "remaining20seeds"]
        self.computecanada = not any(host in self.hostname for host in self.localhosts)

        # experiment parameters

        self.parameters = None

        if not args.resume:

            self.load_data_dir = args.file

            self.load_parameters()
            self.parameters["ab_env_name"] = args.ab_env_name  # addition
            self.parameters["ab_time_steps"] = args.ab_time_steps  # addition
            self.parameters["clear_memory"] = args.clear_memory  # addition
            self.parameters["reinitialize_networks"] = args.reinitialize_networks  # addition
            self.parameters["cuda"] = args.cuda  # update
            self.parameters["file"] = args.file  # addition
            self.parameters["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"  # update

            self.parameters["resumable"] = args.resumable  # update
            self.parameters["resume"] = args.resume  # update
            self.parameters["resume_file"] = args.resume_file  # update
            self.parameters["complete"] = False  # update

        else:

            self.load_data_dir = args.resume_file

            # load old parameters to make sure we are running the same experiment
            self.load_parameters()

            # overwrite parameters that need to be updated
            if not self.parameters["linear_lr_decay"]:  # ONLY experiments with no linear lr decay are properly resumable for a longer number of time steps than originally passed
                self.parameters["ab_time_steps"] = args.ab_time_steps  # overwrite
            self.parameters["cuda"] = args.cuda  # update
            self.parameters["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"  # update

            self.parameters["resumable"] = args.resumable  # update
            self.parameters["resume"] = args.resume  # update
            self.parameters["resume_file"] = args.resume_file  # update
            self.parameters["complete"] = False  # update

        # experiment data directory

        suffix = self.parameters["ab_env_name"] + ":" + str(self.parameters["ab_time_steps"]) \
                 + "_" + self.parameters["n_env_name"] + ":" + str(self.parameters["n_time_steps"]) \
                 + "_lr:" + str(self.parameters["lr"]) \
                 + "_lrd:" + str(self.parameters["linear_lr_decay"]) \
                 + "_slrd:" + str(self.parameters["slow_lrd"]) \
                 + "_g:" + str(self.parameters["gamma"]) \
                 + "_ns:" + str(self.parameters["num_samples"]) \
                 + "_mbs:" + str(self.parameters["mini_batch_size"]) \
                 + "_epo:" + str(self.parameters["epochs"]) \
                 + "_eps:" + str(self.parameters["epsilon"]) \
                 + "_c1:" + str(self.parameters["vf_loss_coef"]) \
                 + "_c2:" + str(self.parameters["policy_entropy_coef"]) \
                 + "_cvl:" + str(self.parameters["clipped_value_fn"]) \
                 + "_mgn:" + str(self.parameters["max_grad_norm"]) \
                 + "_gae:" + str(self.parameters["use_gae"]) \
                 + "_lam:" + str(self.parameters["gae_lambda"]) \
                 + "_hd:" + str(self.parameters["hidden_dim"]) \
                 + "_lstd:" + str(self.parameters["log_std"]) \
                 + "_tef:" + str(self.parameters["time_step_eval_frequency"]) \
                 + "_ee:" + str(self.parameters["eval_episodes"]) \
                 + "_tmsf:" + str(self.parameters["time_step_model_save_frequency"]) \
                 + "_cm:" + str(self.parameters["clear_memory"]) \
                 + "_rn:" + str(self.parameters["reinitialize_networks"]) \
                 + "_d:" + str(self.parameters["device"]) \
                 + ("_r" if self.parameters["resumable"] else "") \
                 + ("_resumed" if self.parameters["resume"] else "")

        self.experiment = "PPOv2_" + suffix

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

        # data

        # data is loaded in call to self.load(), after rl problem is initialized
        self.eval_data = None
        # self.train_data = None
        self.loss_data = None

        # seeds

        # seeds loaded in call to self.load(), after rl problem is initialized
        # Note: we load the seeds after all rl problem elements are created because the creation of the agent
        # network(s) uses xavier initialization, thereby altering the torch seed state

        # env is seeded in Environment __init__() method

        # rl problem

        # abnormal environment used for training
        self.env = Environment(self.parameters["ab_env_name"],
                               self.parameters["seed"])

        # agent
        self.agent = PPOv2(self.env.env_state_dim(),
                           self.env.env_action_dim(),
                           self.parameters["hidden_dim"],
                           self.parameters["log_std"],
                           self.parameters["lr"],
                           self.parameters["linear_lr_decay"],
                           self.parameters["slow_lrd"],
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
                           self.parameters["device"],
                           self.loss_data,
                           self.parameters["resume"])

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = None

        # resume experiment - load data, seed state, env, agent, and rlg
        self.load()

        # clear the memory if indicated by argument
        if not self.parameters["resume"] and self.parameters["clear_memory"]:
            self.rlg.rl_agent_message("clear_memory")

        # reinitialize the networks if indicated by argument
        if not self.parameters["resume"] and self.parameters["reinitialize_networks"]:
            self.rlg.rl_agent_message("reinitialize_networks")

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

        print("abnormal environment name:", highlight_non_default_values("ab_env_name"))
        print("abnormal time steps:", highlight_non_default_values("ab_time_steps"))
        print("normal environment name:", self.parameters["n_env_name"])
        print("normal time steps:", self.parameters["n_time_steps"])
        print("lr:", self.parameters["lr"])
        print("linear lr decay:", self.parameters["linear_lr_decay"])
        print("slow linear lr decay:", self.parameters["slow_lrd"])
        print("gamma:", self.parameters["gamma"])
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
        print("hidden dimension:", self.parameters["hidden_dim"])
        print("log_std:", self.parameters["log_std"])
        print("time step evaluation frequency:", self.parameters["time_step_eval_frequency"])
        print("evaluation episodes:", self.parameters["eval_episodes"])
        print("time step model save frequency:", self.parameters["time_step_model_save_frequency"])
        print("clear memory:", highlight_non_default_values("clear_memory"))
        print("reinitialize networks:", highlight_non_default_values("reinitialize_networks"))
        if self.parameters["device"] == "cuda":
            print("device:", self.parameters["device"])
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                print("cuda visible device(s):", colored(os.environ["CUDA_VISIBLE_DEVICES"], "red"))
            else:
                print(colored("cuda visible device(s): N/A", "red"))
        else:
            print("device:", colored(self.parameters["device"], "red"))
        print("seed:", colored(self.parameters["seed"], "red"))
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
            self.rlg.rl_agent_message("save_model, {}, {}".format(self.data_dir, self.parameters["n_time_steps"]))  # not needed as we already have this model saved
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
            # this is used to ensure that once self.parameters["n_time_steps"] + self.parameters["ab_time_steps"] is reached, the experiment is terminated
            max_steps_this_episode = min(1000, self.parameters["n_time_steps"] + self.parameters["ab_time_steps"] - self.rlg.num_steps())

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

            # index = self.rlg.num_episodes() - 1
            # self.train_data[index] = [self.rlg.num_episodes(), self.rlg.num_steps(), self.rlg.episode_reward()]

            # learning complete
            if self.rlg.num_steps() == self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]:
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

        # if not self.computecanada:
        #     self.send_email(run_time)

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

            if self.parameters["clear_memory"] and self.parameters["reinitialize_networks"]:
                num_updates = ((num_time_steps - self.parameters["n_time_steps"]) // self.parameters["num_samples"])
            elif not self.parameters["clear_memory"] and self.parameters["reinitialize_networks"]:
                num_updates = (num_time_steps - self.parameters["n_time_steps"] + self.agent.memory_init_samples) // self.parameters["num_samples"]
            elif self.parameters["clear_memory"] and not self.parameters["reinitialize_networks"]:
                num_updates = ((num_time_steps - self.parameters["n_time_steps"]) // self.parameters["num_samples"]) + (self.parameters["n_time_steps"] // self.parameters["num_samples"])
            else:
                num_updates = num_time_steps // self.parameters["num_samples"]

            num_epoch_updates = num_updates * self.parameters["epochs"]
            num_mini_batch_updates = num_epoch_updates * (self.parameters["num_samples"] // self.parameters["mini_batch_size"])

            num_samples = num_mini_batch_updates * self.parameters["mini_batch_size"]

            real_time = int(time.time() - self.start)

            index = (num_time_steps // self.parameters["time_step_eval_frequency"]) + 1  # add 1 because we evaluate policy before learning
            self.eval_data[index] = [num_time_steps, num_updates, num_epoch_updates, num_mini_batch_updates, num_samples, average_return, real_time]

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
        Load agent data: models, number of updates of models, and memory.
        Load rlg statistics: num_episodes, num_steps, and total_reward.
        """

        self.load_seed_state()

        self.load_data()

        self.rlg.rl_env_message("load, {}".format(self.load_data_dir))  # load environment data

        if not self.parameters["resume"]:
            self.rlg.rl_agent_message("load, {}, {}".format(self.load_data_dir, self.parameters["n_time_steps"]))  # load agent data
            self.rlg.rl_agent_message("reset_lr")  # reset learning rate to its full value
        else:
            self.rlg.rl_agent_message("load, {}, {}".format(self.load_data_dir, self.parameters["completed_time_steps"]))
        self.agent.loss_data = self.loss_data

        self.load_rlg_statistics()  # load rlg data

    def load_data(self):
        """
        Load experiment data.

        File format: .csv
        """

        csv_foldername = self.load_data_dir + "/csv"

        if not args.resume:

            num_rows = int(self.parameters["ab_time_steps"] / self.parameters["time_step_eval_frequency"]) + 1  # add 1 for evaluation before any learning (0th entry)
            num_columns = 7
            self.eval_data = pd.read_csv(csv_foldername + "/eval_data.csv").to_numpy().copy()[:, 1:]
            self.eval_data = np.append(self.eval_data, np.zeros((num_rows, num_columns)), axis=0)

            # num_rows = self.parameters["ab_time_steps"]  # always larger than needed; will remove extra entries later
            # num_columns = 3
            # self.train_data = pd.read_csv(csv_foldername + "/train_data.csv").to_numpy().copy()[:, 1:]
            # self.train_data = np.append(self.train_data, np.zeros((num_rows, num_columns)), axis=0)

            if self.parameters["clear_memory"]:
                num_rows = self.parameters["ab_time_steps"] // self.parameters["num_samples"]
            else:
                num_rows = ((self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]) // self.parameters["num_samples"]) - (self.parameters["n_time_steps"] // self.parameters["num_samples"])
            num_columns = 8
            self.loss_data = pd.read_csv(csv_foldername + "/loss_data.csv").to_numpy().copy()[:, 1:]
            self.loss_data = np.append(self.loss_data, np.zeros((num_rows, num_columns)), axis=0)

        else:

            self.eval_data = pd.read_csv(csv_foldername + "/eval_data.csv").to_numpy().copy()[:, 1:]
            num_rows = ((self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]) // self.parameters["time_step_eval_frequency"]) + 2 - self.eval_data.shape[0]
            num_columns = self.eval_data.shape[1]
            if num_rows > 0:
                self.eval_data = np.append(self.eval_data, np.zeros((num_rows, num_columns)), axis=0)

            # self.train_data = pd.read_csv(csv_foldername + "/train_data.csv").to_numpy().copy()[:, 1:]
            # num_rows = (self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]) - self.train_data.shape[0]  # always larger than needed; will remove extra entries later
            # num_columns = self.train_data.shape[1]
            # if num_rows > 0:
            #     self.train_data = np.append(self.train_data, np.zeros((num_rows, num_columns)), axis=0)

            self.loss_data = pd.read_csv(csv_foldername + "/loss_data.csv").to_numpy().copy()[:, 1:]
            if self.parameters["clear_memory"]:
                num_rows = (self.parameters["n_time_steps"] // self.parameters["num_samples"]) + (self.parameters["ab_time_steps"] // self.parameters["num_samples"]) - self.loss_data.shape[0]
            else:
                num_rows = ((self.parameters["n_time_steps"] + self.parameters["ab_time_steps"]) // self.parameters["num_samples"]) - self.loss_data.shape[0]
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
        plt.axvline(x=self.parameters["n_time_steps"], ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("time_steps")
        plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Policy Evaluation")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/evaluation_time_steps.jpg")
        plt.close()

        # evaluation: average_return vs num_updates
        df.plot(x="num_updates", y="average_return", color="blue", legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("updates")
        plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Policy Evaluation")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/evaluation_updates.jpg")
        plt.close()

        # evaluation: average_return vs num_samples
        df.plot(x="num_samples", y="average_return", color="blue", legend=False)
        plt.axvline(x=((self.parameters["n_time_steps"] // self.parameters["num_samples"]) * self.parameters["epochs"] * (self.parameters["num_samples"] // self.parameters["mini_batch_size"]) * self.parameters["mini_batch_size"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("samples")
        plt.ylabel("average\nreturn", rotation="horizontal", labelpad=30)
        plt.title("Policy Evaluation")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/evaluation_samples.jpg")
        plt.close()

        # df = pd.read_csv(csv_foldername + "/train_data.csv")
        #
        # # training: episode_return vs num_episodes
        # df.plot(x="num_episodes", y="episode_return", color="blue", legend=False)
        # plt.xlabel("episodes")
        # plt.ylabel("episode\nreturn", rotation="horizontal", labelpad=30)
        # plt.title("Training")
        # pss.plot_settings()
        # plt.savefig(jpg_foldername + "/train_episodes.jpg")
        # plt.close()

        df = pd.read_csv(csv_foldername + "/loss_data.csv")

        # training: clip_loss vs num_updates
        df.plot(x="num_updates", y="clip_loss", color="blue", legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("CLIP Loss")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/clip_loss_updates.jpg")
        plt.close()

        # training: vf_loss vs num_updates
        df.plot(x="num_updates", y="vf_loss", color="blue", legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("VF Loss")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/vf_loss_updates.jpg")
        plt.close()

        # training: entropy vs num_updates
        df.plot(x="num_updates", y="entropy", color="blue", legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("updates")
        plt.ylabel("entropy", rotation="horizontal", labelpad=30)
        plt.title("Entropy")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/entropy_updates.jpg")
        plt.close()

        # training: clip_vf_s_loss vs num_updates
        df.plot(x="num_updates", y="clip_vf_s_loss", color="blue", legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("updates")
        plt.ylabel("loss", rotation="horizontal", labelpad=30)
        plt.title("CLIP+VF+S Loss")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/clip_vf_s_loss_updates.jpg")
        plt.close()

        # training: clip_fraction vs num_updates
        df.plot(x="num_updates", y="clip_fraction", color="blue", legend=False)
        plt.axvline(x=(self.parameters["n_time_steps"] // self.parameters["num_samples"]), ymin=0, ymax=1, color="red", linewidth=2, alpha=0.3)  # malfunction marker
        plt.xlabel("updates")
        plt.ylabel("clip fraction", rotation="horizontal", labelpad=30)
        plt.title("Clip Fraction")
        pss.plot_settings()
        plt.savefig(jpg_foldername + "/clip_fraction_updates.jpg")
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
        Save agent data: models, number of updates of models, and memory.
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
                                     "num_epoch_updates": self.eval_data[:, 2],
                                     "num_mini_batch_updates": self.eval_data[:, 3],
                                     "num_samples": self.eval_data[:, 4],
                                     "average_return": self.eval_data[:, 5],
                                     "real_time": self.eval_data[:, 6]})
        eval_data_df.to_csv(csv_foldername + "/eval_data.csv", float_format="%f")

        # # remove zero entries
        # index = None
        # for i in range(self.train_data.shape[0]):
        #     if (self.train_data[i] == np.zeros(3)).all() and (self.train_data[i+1] == np.zeros(3)).all():
        #         index = i
        #         break
        # self.train_data = self.train_data[:index]
        # train_data_df = pd.DataFrame({"num_episodes": self.train_data[:, 0],
        #                               "num_time_steps": self.train_data[:, 1],
        #                               "episode_return": self.train_data[:, 2]})
        # train_data_df.to_csv(csv_foldername + "/train_data.csv", float_format="%f")

        loss_data_df = pd.DataFrame({"num_updates": self.loss_data[:, 0],
                                     "num_epoch_updates": self.loss_data[:, 1],
                                     "num_mini_batch_updates": self.loss_data[:, 2],
                                     "clip_loss": self.loss_data[:, 3],
                                     "vf_loss": self.loss_data[:, 4],
                                     "entropy": self.loss_data[:, 5],
                                     "clip_vf_s_loss": self.loss_data[:, 6],
                                     "clip_fraction": self.loss_data[:, 7]})
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

    ac = AbnormalController()

    try:

        ac.run()

    except KeyboardInterrupt as e:

        print("keyboard interrupt")


if __name__ == "__main__":

    main()
