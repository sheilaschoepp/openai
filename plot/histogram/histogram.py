import argparse
import os
import pickle
import random
import xml.etree.ElementTree as ET
from pathlib import Path

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

parser.add_argument("-s", "--num_seeds", type=int, default=30, metavar="N",
                    help="collect histogram data across s seeds (default: 30)")

parser.add_argument("-cd", "--collect_data", default=False, action="store_true",
                    help="if True, collect histogram data by running the policy for 100 episodes (default: False)")

args = parser.parse_args()


def plot_ant_histograms():
    """
    Plot Ant histograms.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]

    format: .jpg
    """

    histogram_plot_directory = os.getcwd() + "/plots/ant/{}/{}".format(algorithm, env_name)
    os.makedirs(histogram_plot_directory, exist_ok=True)

    df = pd.DataFrame(ant_histogram_data[0], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(-0.6, 0.6, 0.1)).set_title("{}, {}: hip_1".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_hip_1_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.show()
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

    histogram_data_directory = os.getcwd() + "/data/ant/{}/{}".format(env_name, algorithm)
    os.makedirs(histogram_data_directory, exist_ok=True)

    np.save(histogram_data_directory + "/{}_{}_histogram_data_{}.npy".format(env_name, algorithm, args.num_seeds), ant_histogram_data)


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

        for _ in range(100):

            state, _ = self.rlg.rl_start()
            save_ant_joint_angles(state)

            terminal = False

            max_steps_this_episode = 1000
            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, state, terminal, _ = self.rlg.rl_step()
                save_ant_joint_angles(state)


def get_fetchrach_xml_data():

    if "melco2" in os.uname()[1]:
        anaconda_path = "/opt/anaconda3"
    elif "melco" in os.uname()[1]:
        anaconda_path = "/local/melco2/sschoepp/anaconda3"
    else:
        anaconda_path = os.getenv("HOME") + "/anaconda3"

    model_xml = None
    if env_name == "FetchReach-v1":
        model_xml = anaconda_path + "/envs/openai3.7/lib/python3.7/site-packages/gym/envs/robotics/assets/fetch/reach.xml"  # todo
    if "v0" in env_name:
        model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"
    elif "v1" in env_name:
        model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/reach.xml"
    elif "v2" in env_name:
        model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/reach.xml"
    elif "v3" in env_name:
        model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/reach.xml"
    elif "v4" in env_name:
        model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenShoulderLiftSensor/assets/fetch/reach.xml"
    elif "v5" in env_name:
        model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v5_BrokenJointsTBD/assets/fetch/reach.xml"

    robot_xml = model_xml[:-9] + "robot.xml"
    tree = ET.parse(robot_xml)
    root = tree.getroot()

    shoulder_pan_joint_range = None
    shoulder_lift_joint_range = None
    upperarm_roll_joint_range = None
    elbow_flex_joint_range = None
    forearm_roll_joint_range = None
    wrist_flex_joint_range = None
    wrist_roll_joint_range = None

    for child in root.iter():
        attrib = child.attrib
        name = attrib.get("name")
        if name == "robot0:torso_lift_joint":
            torso_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:head_pan_joint":
            head_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:head_tilt_joint":
            head_tilt_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:shoulder_pan_joint":
            shoulder_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:shoulder_lift_joint":
            shoulder_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:upperarm_roll_joint":
            if attrib.get("range") is not None:
                upperarm_roll_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            else:
                upperarm_roll_joint_range = np.array([-3.14, 3.14])
        elif name == "robot0:elbow_flex_joint":
            elbow_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:forearm_roll_joint":
            if attrib.get("range") is not None:
                forearm_roll_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            else:
                forearm_roll_joint_range = np.array([-3.14, 3.14])
        elif name == "robot0:wrist_flex_joint":
            wrist_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:wrist_roll_joint":
            if attrib.get("range") is not None:
                wrist_roll_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            else:
                wrist_roll_joint_range = np.array([-3.14, 3.14])
        elif name == "robot0:r_gripper_finger_joint":
            r_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
        elif name == "robot0:l_gripper_finger_joint":
            l_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)

    return shoulder_pan_joint_range, shoulder_lift_joint_range, upperarm_roll_joint_range, elbow_flex_joint_range, forearm_roll_joint_range, wrist_flex_joint_range, wrist_roll_joint_range


def plot_fetchreach_histograms(ranges):
    """
    Load data and plots FetchReach histograms.

    histogram data:
    [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]

    format: .jpg

    @param ranges: tuple of numpy arrays
        joint ranges
    """
    global fetchreach_histogram_data

    histogram_data_directory = os.getcwd() + "/data/fetchreach/{}/{}".format(algorithm, env_name)
    fetchreach_histogram_data = np.load(histogram_data_directory + "/{}_{}_histogram_data_{}.npy".format(algorithm, env_name, args.num_seeds))

    histogram_plot_directory = os.getcwd() + "/plots/fetchreach/{}/{}".format(algorithm, env_name)
    os.makedirs(histogram_plot_directory, exist_ok=True)

    index = 0  # shoulder_pan_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:shoulder_pan_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_shoulder_pan_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()

    index = 1  # shoulder_lift_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:shoulder_lift_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_shoulder_lift_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()

    index = 2  # upperarm_roll_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:upperarm_roll_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_upperarm_roll_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()

    index = 3  # elbow_flex_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:elbow_flex_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_elbow_flex_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()

    index = 4  # forearm_roll_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:forearm_roll_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_forearm_roll_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()

    index = 5  # wrist_flex_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:wrist_flex_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_wrist_flex_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()

    index = 6  # wrist_roll_joint
    df = pd.DataFrame(fetchreach_histogram_data[index], columns=["radians"])
    sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.1)).set_title("{}, {}: robot0:wrist_roll_joint".format(algorithm, env_name), fontweight="bold")
    plt.savefig(histogram_plot_directory + "/{}_{}_wrist_roll_joint_{}.jpg".format(algorithm, env_name, args.num_seeds))
    plt.close()


def plot_fetchreach_heatmap(ranges):
    """
    Plot FetchReach heatmap.

    histogram data:
    [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]

    format: .jpg

    @param ranges: tuple of numpy arrays
        joint ranges
    """
    global fetchreach_histogram_data

    histogram_data_directory = os.getcwd() + "/data/fetchreach/{}/{}".format(algorithm, env_name)
    fetchreach_histogram_data = np.load(histogram_data_directory + "/{}_{}_histogram_data_{}.npy".format(algorithm, env_name, args.num_seeds))

    fetchreach_histogram_data_normalized = []

    # normalize data
    for i in range(len(ranges)):

        min = ranges[i][0]
        max = ranges[i][1]
        data = fetchreach_histogram_data[i]

        normalized = (data - min) / (max - min)

        fetchreach_histogram_data_normalized.append(normalized)

    fetchreach_histogram_count_data = []
    bins = np.round(np.arange(0.0, 1.025, 0.025), 3)

    for i in range(7):
        counts, _ = np.histogram(fetchreach_histogram_data_normalized[i], bins=bins, range=[0.0, 1.0])
        counts = counts / np.sum(counts)  # probability
        fetchreach_histogram_count_data.append(counts)

    df = pd.DataFrame(np.array(fetchreach_histogram_count_data).T, index=bins[:-1], columns=["shoulder pan", "shoulder lift", "upperarm roll", "elbow flex", "forearm roll", "wrist flex", "wrist roll"])

    histogram_plot_directory = os.getcwd() + "/plots/fetchreach/{}/{}".format(algorithm, env_name)
    os.makedirs(histogram_plot_directory, exist_ok=True)

    heatmap = sns.heatmap(data=df, cmap="viridis")
    heatmap.set_title("{}, {}: Visited Joint Angles".format(algorithm, env_name), fontweight="bold")
    plt.xlabel("joint")
    plt.ylabel("normalized angle")
    plt.tight_layout()
    plt.savefig(histogram_plot_directory + "/{}_{}_heatmap_{}.jpg".format(algorithm, env_name, args.num_seeds))


def save_fetchreach_histogram_data():
    """
    Save FetchReach histogram data.

    format: .npy and .pkl

    Note: pkl file has joint labels.
    """

    histogram_data_directory = os.getcwd() + "/data/fetchreach/{}/{}".format(algorithm, env_name)
    os.makedirs(histogram_data_directory, exist_ok=True)

    np.save(histogram_data_directory + "/{}_{}_histogram_data_{}.npy".format(algorithm, env_name, args.num_seeds), fetchreach_histogram_data)

    df = pd.DataFrame(np.array(fetchreach_histogram_data).T, columns=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"])
    df.to_pickle(histogram_data_directory + "/{}_{}_histogram_data_{}.pkl".format(algorithm, env_name, args.num_seeds))


def save_fetchreach_joint_angles(d):
    """
    Save FetchReach visited joint angles

    histogram data:
    [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]

    @param d: PyMjData object (https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data)
        mujoco-py simulation data
    """

    fetchreach_histogram_data[0].append(d.get_joint_qpos("robot0:shoulder_pan_joint"))
    fetchreach_histogram_data[1].append(d.get_joint_qpos("robot0:shoulder_lift_joint"))
    fetchreach_histogram_data[2].append(d.get_joint_qpos("robot0:upperarm_roll_joint"))
    fetchreach_histogram_data[3].append(d.get_joint_qpos("robot0:elbow_flex_joint"))
    fetchreach_histogram_data[4].append(d.get_joint_qpos("robot0:forearm_roll_joint"))
    fetchreach_histogram_data[5].append(d.get_joint_qpos("robot0:wrist_flex_joint"))
    fetchreach_histogram_data[6].append(d.get_joint_qpos("robot0:wrist_roll_joint"))


def collect_fetchreach_data():

    pbar = tqdm(total=args.num_seeds)

    for seed in range(args.num_seeds):
        pbar.set_description("Processing seed {}".format(seed))

        sim = FetchReachHistogram(seed)
        sim.run()
        sim.cleanup()

        pbar.update(1)

    save_fetchreach_histogram_data()


def plot_fetchreach_data():

    ranges = get_fetchrach_xml_data()

    plot_fetchreach_histograms(ranges)

    plot_fetchreach_heatmap(ranges)


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
            self.env_name = self.parameters["ab_env_name"]
        else:
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

    env_name = args.file.split("_")[1].split(":")[0]

    algorithm = args.file.split("_")[0].split("/")[-1][:-2]

    if "Ant" in env_name:

        # histogram data:
        # [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]
        ant_histogram_data = [[], [], [], [], [], [], [], []]

        # TODO

    elif "FetchReach" in env_name:

        # histogram data:
        # [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]
        fetchreach_histogram_data = [[], [], [], [], [], [], []]

        if args.collect_data:

            collect_fetchreach_data()

        plot_fetchreach_data()

    else:

        print("file argument does not include Ant or FetchReach")
