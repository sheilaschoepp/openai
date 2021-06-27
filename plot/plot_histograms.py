import argparse
import math
import os
import pickle
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from controllers.ppov2.ppov2_agent import PPOv2
from controllers.sacv2.sacv2_agent import SACv2
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs  # DO NOT DELETE

sns.set_theme()

parser = argparse.ArgumentParser(description="Simulate Arguments")

parser.add_argument("-f", "--file", default="",
                    help="absolute path of the folder containing data for all seeds")

parser.add_argument("-t", "--time_steps", default="",
                    help="the number of time steps into learning")

args = parser.parse_args()


NUM_SEEDS = 30
NUM_EPISODES_PER_SEED = 100


def plot_ant_histograms():
    """
    Plot Ant histograms.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]

    format: .jpg
    """

    histogram_plot_directory = os.getcwd() + "/plotted_histogram_results/ant/{}/{}".format(env_name, algorithm)
    os.makedirs(histogram_plot_directory, exist_ok=True)

    # df = pd.DataFrame(fetchreach_histogram_data[0], columns=["radians"])
    # sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(0, 0.5, 0.01))
    # plt.savefig(histogram_plot_directory + "/{}_{}_torso_lift_joint.jpg".format(env_name, algorithm))
    # plt.close()

    # pss.plot_settings()
    # plt.hist(ant_histogram_data[0])
    # plt.title("hip_1")
    # plt.axvline(x=-np.radians(30), color="red")
    # plt.axvline(x=np.radians(30), color="red")
    # plt.savefig(histogram_directory + "/hip_1.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[1])
    # plt.title("ankle_1")
    # plt.axvline(x=np.radians(30), color="red")
    # plt.axvline(x=np.radians(70), color="red")
    # plt.savefig(histogram_directory + "/ankle_1.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[2])
    # plt.title("hip_2")
    # plt.axvline(x=-np.radians(30), color="red")
    # plt.axvline(x=np.radians(30), color="red")
    # plt.savefig(histogram_directory + "/hip_2.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[3])
    # plt.title("ankle_2")
    # plt.axvline(x=-np.radians(70), color="red")
    # plt.axvline(x=-np.radians(30), color="red")
    # plt.savefig(histogram_directory + "/ankle_2.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[4])
    # plt.title("hip_3")
    # plt.axvline(x=-np.radians(30), color="red")
    # plt.axvline(x=np.radians(30), color="red")
    # plt.savefig(histogram_directory + "/hip_3.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[5])
    # plt.title("ankle_3")
    # plt.axvline(x=-np.radians(70), color="red")
    # plt.axvline(x=-np.radians(30), color="red")
    # plt.savefig(histogram_directory + "/ankle_3.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[6])
    # plt.title("hip_4")
    # plt.axvline(x=-np.radians(30), color="red")
    # plt.axvline(x=np.radians(30), color="red")
    # plt.savefig(histogram_directory + "/hip_4.jpg")
    # plt.clf()
    #
    # pss.plot_settings()
    # plt.hist(ant_histogram_data[7])
    # plt.title("ankle_4")
    # plt.axvline(x=np.radians(30), color="red")
    # plt.axvline(x=np.radians(70), color="red")
    # plt.savefig(histogram_directory + "/ankle_4.jpg")
    # plt.clf()


def save_ant_histogram_data():
    """
    Save Ant histogram data.

    format: .npy
    """

    histogram_data_directory = os.getcwd() + "/numerical_histogram_results/ant/{}/{}".format(env_name, algorithm)
    os.makedirs(histogram_data_directory, exist_ok=True)

    np.save(histogram_data_directory + "/{}_{}_histogram_data.npy".format(env_name, algorithm), ant_histogram_data)


def save_ant_joint_angles(s):
    """
    Save Ant visited joint angles.

    @param s: float64 numpy array
        state of the environment
    """

    ant_histogram_data[0].append(s[5])
    ant_histogram_data[1].append(s[6])
    ant_histogram_data[2].append(s[7])
    ant_histogram_data[3].append(s[8])
    ant_histogram_data[4].append(s[9])
    ant_histogram_data[5].append(s[10])
    ant_histogram_data[6].append(s[11])
    ant_histogram_data[7].append(s[12])


class AntHistogram:
    """
    Controller for creating a histogram of visited joint angles for Ant.
    """

    LINE = "--------------------------------------------------------------------------------"

    def __init__(self, seed):
        """
        @param seed: int
            experiment seed
        """

        self.load_data_dir = args.file + "/seed{}".format(seed)
        self.t = int(args.time_steps)

        self.parameters = None
        self.load_parameters()

        if "ab_env_name" in self.parameters:
            print("loading abnormal environment")
            self.env_name = self.parameters["ab_env_name"]
        else:
            print("loading normal environment")
            self.env_name = self.parameters["n_env_name"]

        # seeds

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.parameters["device"] == "cuda":
            torch.cuda.manual_seed(seed)

        # env is seeded in Environment __init__() method

        # rl problem

        # normal environment used for training
        self.env = Environment(self.env_name,
                               seed,
                               render=False)

        # agent
        if "SAC" in self.load_data_dir:

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
                               None)

        elif "PPO" in self.load_data_dir:

            self.agent = PPOv2(self.env.env_state_dim(),
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
                               self.parameters["device"],
                               None,
                               False)

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = {"num_episodes": 0, "num_steps": 0, "total_reward": 0}

        # load agent model

        self.rlg.rl_agent_message("load_model, {}, {}".format(self.load_data_dir, self.t))  # load agent data
        self.rlg.rl_agent_message("mode, eval")  # set mode to eval: no learning, deterministic policy

    def cleanup(self):
        """
        Close environment.
        """

        self.rlg.rl_env_message("close")

    def load_parameters(self):
        """
        Load normal experiment parameters.

        File format: .pickle
        """

        pickle_foldername = self.load_data_dir + "/pickle"

        with open(pickle_foldername + "/parameters.pickle", "rb") as f:
            self.parameters = pickle.load(f)

    def run(self):

        self.rlg.rl_init(total_reward=self.rlg_statistics["total_reward"],
                         num_steps=self.rlg_statistics["num_steps"],
                         num_episodes=self.rlg_statistics["num_episodes"])

        for _ in range(NUM_EPISODES_PER_SEED):

            state, _ = self.rlg.rl_start()
            save_ant_joint_angles(state)

            terminal = False

            max_steps_this_episode = 1000
            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, state, terminal, _ = self.rlg.rl_step()
                save_ant_joint_angles(state)


def plot_fetchreach_histograms():
    """
    Plot FetchReach histograms.

    histogram data:
    [torso_lift_joint, head_pan_joint, head_tilt_joint, shoulder_pan_joint, shoulder_lift_joint, elbow_flex_joint, wrist_flex_joint, r_gripper_finger_joint, l_gripper_finger_joint]

    format: .jpg
    """

    histogram_plot_directory = os.getcwd() + "/plotted_histogram_results/fetchreach/{}/{}".format(env_name, algorithm)
    os.makedirs(histogram_plot_directory, exist_ok=True)

    df = pd.DataFrame(fetchreach_histogram_data[0], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(0, 0.5, 0.01))
    plt.savefig(histogram_plot_directory + "/{}_{}_torso_lift_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[1], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability",  bins=np.arange(-1.6, 1.7, 0.1))
    plt.savefig(histogram_plot_directory + "/{}_{}_head_pan_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[2], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(-0.7, 1.6, 0.1))
    plt.savefig(histogram_plot_directory + "/{}_{}_head_tilt_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[3], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(-1.7, 1.8, 0.1))
    plt.savefig(histogram_plot_directory + "/{}_{}_shoulder_pan_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[4], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(-1.3, 1.7, 0.1))
    plt.savefig(histogram_plot_directory + "/{}_{}_shoulder_lift_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[5], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(-2.3, 2.4, 0.1))
    plt.savefig(histogram_plot_directory + "/{}_{}_elbow_flex_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[6], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(-2.2, 2.3, 0.1))
    plt.savefig(histogram_plot_directory + "/{}_{}_wrist_flex_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[7], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(0.0, 0.6, 0.01))
    plt.savefig(histogram_plot_directory + "/{}_{}_r_gripper_finger_joint.jpg".format(env_name, algorithm))
    plt.close()

    df = pd.DataFrame(fetchreach_histogram_data[8], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:brown", stat="probability", bins=np.arange(0.0, 0.6, 0.01))
    plt.savefig(histogram_plot_directory + "/{}_{}_l_gripper_finger_joint.jpg".format(env_name, algorithm))
    plt.close()


def save_fetchreach_histogram_data():
    """
    Save FetchReach histogram data.

    format: .npy
    """

    histogram_data_directory = os.getcwd() + "/numerical_histogram_results/fetchreach/{}/{}".format(env_name, algorithm)
    os.makedirs(histogram_data_directory, exist_ok=True)

    np.save(histogram_data_directory + "/{}_{}_histogram_data.npy".format(env_name, algorithm), fetchreach_histogram_data)


def save_fetchreach_joint_angles(d):
    """
    Save FetchReach visited joint angles

    @param d: PyMjData object (https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data)
        mujoco-py simulation data
    """

    fetchreach_histogram_data[0].append(d.get_joint_qpos("robot0:torso_lift_joint"))
    fetchreach_histogram_data[1].append(d.get_joint_qpos("robot0:head_pan_joint"))
    fetchreach_histogram_data[2].append(d.get_joint_qpos("robot0:head_tilt_joint"))
    fetchreach_histogram_data[3].append(d.get_joint_qpos("robot0:shoulder_pan_joint"))
    fetchreach_histogram_data[4].append(d.get_joint_qpos("robot0:shoulder_lift_joint"))
    fetchreach_histogram_data[5].append(d.get_joint_qpos("robot0:elbow_flex_joint"))
    fetchreach_histogram_data[6].append(d.get_joint_qpos("robot0:wrist_flex_joint"))
    fetchreach_histogram_data[7].append(d.get_joint_qpos("robot0:r_gripper_finger_joint"))
    fetchreach_histogram_data[8].append(d.get_joint_qpos("robot0:l_gripper_finger_joint"))


class FetchReachHistogram:
    """
    Controller for creating a histogram of visited joint angles for FetchReach.
    """

    LINE = "--------------------------------------------------------------------------------"

    def __init__(self, seed):
        """
        @param seed: int
            experiment seed
        """

        self.load_data_dir = args.file + "/seed{}".format(seed)
        self.t = int(args.time_steps)

        self.parameters = None
        self.load_parameters()

        if "ab_env_name" in self.parameters:
            # print("loading abnormal environment")
            self.env_name = self.parameters["ab_env_name"]
        else:
            # print("loading normal environment")
            self.env_name = self.parameters["n_env_name"]

        # seeds

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.parameters["device"] == "cuda":
            torch.cuda.manual_seed(seed)

        # env is seeded in Environment __init__() method

        # rl problem

        # normal environment used for training
        self.env = Environment(self.env_name,
                               seed,
                               render=False)

        # agent
        if "SAC" in self.load_data_dir:

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
                               None)

        elif "PPO" in self.load_data_dir:

            self.agent = PPOv2(self.env.env_state_dim(),
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
                               self.parameters["device"],
                               None,
                               False)

        # RLGlue used for training
        self.rlg = RLGlue(self.env, self.agent)
        self.rlg_statistics = {"num_episodes": 0, "num_steps": 0, "total_reward": 0}

        # load agent model

        self.rlg.rl_agent_message("load_model, {}, {}".format(self.load_data_dir, self.t))  # load agent data
        self.rlg.rl_agent_message("mode, eval")  # set mode to eval: no learning, deterministic policy

    def cleanup(self):
        """
        Close environment.
        """

        self.rlg.rl_env_message("close")

    def load_parameters(self):
        """
        Load normal experiment parameters.

        File format: .pickle
        """

        pickle_foldername = self.load_data_dir + "/pickle"

        with open(pickle_foldername + "/parameters.pickle", "rb") as f:
            self.parameters = pickle.load(f)

    def run(self):

        self.rlg.rl_init(total_reward=self.rlg_statistics["total_reward"],
                         num_steps=self.rlg_statistics["num_steps"],
                         num_episodes=self.rlg_statistics["num_episodes"])

        for _ in range(NUM_EPISODES_PER_SEED):

            state, _ = self.rlg.rl_start()
            save_fetchreach_joint_angles(self.env.env.sim.data)

            terminal = False

            max_steps_this_episode = 1000
            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, state, terminal, _ = self.rlg.rl_step()
                save_fetchreach_joint_angles(self.env.env.sim.data)


if __name__ == "__main__":

    env_name = args.file.split("_")[1].split(":")[0]

    algorithm = args.file.split("_")[0].split("/")[-1][:-2]

    if "Ant" in env_name:

        # histogram data:
        # [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]
        ant_histogram_data = [[], [], [], [], [], [], [], []]

        pbar = tqdm(total=NUM_SEEDS)

        for seed in range(NUM_SEEDS):

            pbar.set_description("Processing seed {}".format(seed))

            sim = AntHistogram(seed)
            sim.run()
            sim.cleanup()

            pbar.update(1)

        plot_ant_histograms()

        save_ant_histogram_data()

    elif "FetchReach" in env_name:

        # histogram data:
        # [torso_lift_joint, head_pan_joint, head_tilt_joint, shoulder_pan_joint, shoulder_lift_joint, elbow_flex_joint, wrist_flex_joint, r_gripper_finger_joint, l_gripper_finger_joint]
        fetchreach_histogram_data = [[], [], [], [], [], [], [], [], []]

        pbar = tqdm(total=NUM_SEEDS)

        for seed in range(NUM_SEEDS):

            pbar.set_description("Processing seed {}".format(seed))

            sim = FetchReachHistogram(seed)
            sim.run()
            sim.cleanup()

            pbar.update(1)

        plot_fetchreach_histograms()

        save_fetchreach_histogram_data()

    else:

        print("file argument does not include Ant or FetchReach")
