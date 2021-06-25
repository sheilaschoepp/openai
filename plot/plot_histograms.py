import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.plot_style_settings as pss
from controllers.ppov2.ppov2_agent import PPOv2
from controllers.sacv2.sacv2_agent import SACv2
from environment.environment import Environment
from utils.rl_glue import RLGlue

parser = argparse.ArgumentParser(description="Simulate Arguments")

parser.add_argument("-f", "--file", default="",
                    help="absolute path of the folder containing data for all seeds")

parser.add_argument("-t", "--time_steps", default="",
                    help="the number of time steps into learning")

args = parser.parse_args()


def plot_ant_histograms():
    """
    Plot Ant histograms.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]
    """

    histogram_directory = os.getcwd() + "/plotted_histograms/ant"
    os.makedirs(histogram_directory, exist_ok=True)

    pss.plot_settings()
    plt.hist(ant_histogram_data[0])
    plt.title("hip_1")
    plt.axvline(x=-np.radians(30), color="red")
    plt.axvline(x=np.radians(30), color="red")
    plt.savefig(histogram_directory + "/hip_1.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[1])
    plt.title("ankle_1")
    plt.axvline(x=np.radians(30), color="red")
    plt.axvline(x=np.radians(70), color="red")
    plt.savefig(histogram_directory + "/ankle_1.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[2])
    plt.title("hip_2")
    plt.axvline(x=-np.radians(30), color="red")
    plt.axvline(x=np.radians(30), color="red")
    plt.savefig(histogram_directory + "/hip_2.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[3])
    plt.title("ankle_2")
    plt.axvline(x=-np.radians(70), color="red")
    plt.axvline(x=-np.radians(30), color="red")
    plt.savefig(histogram_directory + "/ankle_2.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[4])
    plt.title("hip_3")
    plt.axvline(x=-np.radians(30), color="red")
    plt.axvline(x=np.radians(30), color="red")
    plt.savefig(histogram_directory + "/hip_3.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[5])
    plt.title("ankle_3")
    plt.axvline(x=-np.radians(70), color="red")
    plt.axvline(x=-np.radians(30), color="red")
    plt.savefig(histogram_directory + "/ankle_3.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[6])
    plt.title("hip_4")
    plt.axvline(x=-np.radians(30), color="red")
    plt.axvline(x=np.radians(30), color="red")
    plt.savefig(histogram_directory + "/hip_4.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(ant_histogram_data[7])
    plt.title("ankle_4")
    plt.axvline(x=np.radians(30), color="red")
    plt.axvline(x=np.radians(70), color="red")
    plt.savefig(histogram_directory + "/ankle_4.jpg")
    plt.clf()


def save_ant_joint_angles(s):
    """
    Save the Ant's joint angles.

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

        for _ in range(100):

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
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]
    """

    histogram_directory = os.getcwd() + "/plotted_histograms/fetchreach"
    os.makedirs(histogram_directory, exist_ok=True)

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[0])
    plt.title("torso_lift_joint")
    plt.axvline(x=0.0386, color="red")
    plt.axvline(x=0.3861, color="red")
    plt.savefig(histogram_directory + "/torso_lift_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[1])
    plt.title("head_pan_joint")
    # plt.axvline(x=-1.57, color="red")  # todo: uncomment
    # plt.axvline(x=1.57, color="red")
    plt.savefig(histogram_directory + "/head_pan_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[2])
    plt.title("head_tilt_joint")
    # plt.axvline(x=-0.76, color="red")  # todo: uncomment
    # plt.axvline(x=1.45, color="red")
    plt.savefig(histogram_directory + "/head_tilt_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[3])
    plt.title("shoulder_pan_joint")
    plt.axvline(x=-1.6056, color="red")
    plt.axvline(x=1.6056, color="red")
    plt.savefig(histogram_directory + "/shoulder_pan_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[4])
    plt.title("shoulder_lift_joint")
    plt.axvline(x=-1.221, color="red")
    plt.axvline(x=1.518, color="red")
    plt.savefig(histogram_directory + "/shoulder_lift_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[5])
    plt.title("elbow_flex_joint")
    plt.axvline(x=-2.251, color="red")
    plt.axvline(x=2.251, color="red")
    plt.savefig(histogram_directory + "/elbow_flex_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[6])
    plt.title("wrist_flex_joint")
    plt.axvline(x=-2.16, color="red")
    plt.axvline(x=2.16, color="red")
    plt.savefig(histogram_directory + "/wrist_flex_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[7])
    plt.title("r_gripper_finger_joint")
    # plt.axvline(x=0.0, color="red")  # todo: uncomment
    # plt.axvline(x=0.05, color="red")
    plt.savefig(histogram_directory + "/r_gripper_finger_joint.jpg")
    plt.clf()

    pss.plot_settings()
    plt.hist(fetchreach_histogram_data[8])
    plt.title("l_gripper_finger_joint")
    # plt.axvline(x=0.0, color="red")  # todo: uncomment
    # plt.axvline(x=0.05, color="red")
    plt.savefig(histogram_directory + "/l_gripper_finger_joint.jpg")
    plt.clf()


def save_fetchreach_joint_angles(d):
    """
    Save the FetchReach robot's joint angles.

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

        for _ in range(100):

            state, _ = self.rlg.rl_start()
            save_fetchreach_joint_angles(self.env.env.sim.data)

            terminal = False

            max_steps_this_episode = 1000
            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, state, terminal, _ = self.rlg.rl_step()
                save_fetchreach_joint_angles(self.env.env.sim.data)


if __name__ == "__main__":

    if "Ant" in args.file:

        # histogram data:
        # [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]
        ant_histogram_data = [[], [], [], [], [], [], [], []]

        for seed in range(0, 1):  # todo

            sim = AntHistogram(seed)
            sim.run()

        plot_ant_histograms()

    elif "FetchReach" in args.file:

        # histogram data:
        # [torso_lift_joint, head_pan_joint, head_tilt_joint, shoulder_pan_joint, shoulder_lift_joint, elbow_flex_joint, wrist_flex_joint, r_gripper_finger_joint, l_gripper_finger_joint]
        fetchreach_histogram_data = [[], [], [], [], [], [], [], [], []]

        for seed in range(0, 1):  # todo

            sim = FetchReachHistogram(seed)
            sim.run()

        plot_fetchreach_histograms()

    else:

        print("file argument does not include Ant or FetchReach")
