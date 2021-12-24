import argparse
import os
import pickle
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5 as pickle
import seaborn as sns
import torch
from termcolor import colored
from tqdm import tqdm

from controllers.ppov2.ppov2_agent import PPOv2
from controllers.sacv2.sacv2_agent import SACv2
from environment.environment import Environment
from utils.rl_glue import RLGlue

import custom_gym_envs  # DO NOT DELETE

sns.set_theme()
cmap = "viridis"

LARGE = 16
MEDIUM = 14

plt.rc("axes", titlesize=LARGE)     # fontsize of the axes title
plt.rc("axes", labelsize=LARGE)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM)   # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM)   # fontsize of the tick labels
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"

parser = argparse.ArgumentParser(description="Histogram")

parser.add_argument("-cd", "--collect_data", default=False, action="store_true",
                    help="if true, collect data by running 100 episodes for each seed (default: False)")

args = parser.parse_args()


def get_ant_xml_data():

    if "melco2" in os.uname()[1]:
        anaconda_path = "/opt/anaconda3"
    elif "melco" in os.uname()[1]:
        anaconda_path = "/local/melco2/sschoepp/anaconda3"
    else:
        anaconda_path = os.getenv("HOME") + "/anaconda3"

    # model_xml = None
    # if env_name == "Ant-v2":
    #     model_xml = anaconda_path + "/envs/openai3.7/lib/python3.7/site-packages/gym/envs/mujoco/assets/ant.xml"
    # elif "v0" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/ant/xml/AntEnv_v0_Normal.xml"
    # elif "v1" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/ant/xml/AntEnv_v1_BrokenSeveredLimb.xml"
    # elif "v2" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/ant/xml/AntEnv_v2_Hip4ROM.xml"
    # elif "v3" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/ant/xml/AntEnv_v3_Ankle4ROM.xml"
    # elif "v4" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/ant/xml/AntEnv_v4_BrokenUnseveredLimb.xml"

    # use normal xml to define the ranges for plotting
    model_xml = anaconda_path + "/envs/openai3.7/lib/python3.7/site-packages/gym/envs/mujoco/assets/ant.xml"

    tree = ET.parse(model_xml)
    root = tree.getroot()

    hip_1_range = None
    ankle_1_range = None
    hip_2_range = None
    ankle_2_range = None
    hip_3_range = None
    ankle_3_range = None
    hip_4_range = None
    ankle_4_range = None

    for child in root.iter():
        attrib = child.attrib
        name = attrib.get("name")
        if name == "hip_1":
            hip_1_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "ankle_1":
            ankle_1_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "hip_2":
            hip_2_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "ankle_2":
            ankle_2_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "hip_3":
            hip_3_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "ankle_3":
            ankle_3_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "hip_4":
            hip_4_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))
        elif name == "ankle_4":
            ankle_4_range = np.radians(np.array(attrib.get("range").split(" "), dtype=float))

    return hip_1_range, ankle_1_range, hip_2_range, ankle_2_range, hip_3_range, ankle_3_range, hip_4_range, ankle_4_range


def plot_ant_histograms(ranges):
    """
    Load data and plots Ankle histograms.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]

    format: .jpg

    @param ranges: tuple of numpy arrays
        joint ranges
    """
    global ant_histogram_data

    ant_histogram_data = np.load(experiment_data_directory + "/{}_histogram_data_{}.npy".format(experiment_name, num_seeds))

    ant_histogram_data_clipped = []

    # clip data
    for i in range(len(ranges)):
        min = ranges[i][0]
        max = ranges[i][1]
        data = ant_histogram_data[i]

        clipped = np.clip(data, min, max)

        ant_histogram_data_clipped.append(clipped)

    def title():
        if suffix == "":
            title = "{}\n{}\n{}".format(algorithm, name, joint_name)  # todo
        else:
            title = "{} ({})\n{}\n{}".format(algorithm, suffix_eval, name, joint_name)
        return title

    index = 0  # hip_1
    joint_name = "hip_1"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.02, 0.02), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_hip_1_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 1  # ankle_1
    joint_name = "ankle_1"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.015, 0.015), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_ankle_1_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 2  # hip_2
    joint_name = "hip_2"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.02, 0.02), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_hip_2_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 3  # ankle_2
    joint_name = "ankle_2"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.015, 0.015), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_ankle_2_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 4  # hip_3
    joint_name = "hip_3"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.02, 0.02), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_hip_3_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 5  # ankle_3
    joint_name = "ankle_3"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.015, 0.015), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_ankle_3_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 6  # hip_4
    joint_name = "hip_4"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.02, 0.02), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_hip_4_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 7  # ankle_4
    joint_name = "ankle_4"
    df = pd.DataFrame(ant_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1], 0.015), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_ankle_4_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()


def plot_ant_heatmap(ranges):
    """
    Plot FetchReach heatmap.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]

    format: .jpg

    @param ranges: tuple of numpy arrays
        joint ranges
    """
    global ant_histogram_data

    ant_histogram_data = np.load(experiment_data_directory + "/{}_histogram_data_{}.npy".format(experiment_name, num_seeds))

    ant_histogram_data_normalized = []

    # normalize and clip data
    for i in range(len(ranges)):

        min = ranges[i][0]
        max = ranges[i][1]
        data = ant_histogram_data[i]

        normalized = (data - min) / (max - min)

        normalized = np.clip(normalized, 0, 1)

        ant_histogram_data_normalized.append(normalized)

    ant_histogram_count_data = []
    bins = np.arange(0.0, 1.05, 0.025)

    for i in range(8):
        counts, _ = np.histogram(ant_histogram_data_normalized[i], bins=bins)
        counts = counts / np.sum(counts)  # probability
        ant_histogram_count_data.append(counts)

    df = pd.DataFrame(np.array(ant_histogram_count_data).T, index=bins[:-1], columns=["hip 1", "ankle 1", "hip 2", "ankle 2", "hip 3", "ankle 3", "hip 4", "ankle 4"])

    y_labels = ["1.0", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "0.5", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "0.0"]
    heatmap = sns.heatmap(data=df[::-1], cmap=cmap, yticklabels=y_labels)

    algorithm_ = None

    if algorithm == "PPO":
        algorithm_ = "Proximal Policy Optimization (PPO)"
    elif algorithm == "SAC":
        algorithm_ = "Soft Actor-Critic (SAC)"

    if suffix == "":
        heatmap.set_title("{}".format(algorithm_))
    else:
        heatmap.set_title("{} ({})\n{}".format(algorithm, suffix_eval, name))
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel("joint")
    plt.ylabel("normalized angle")
    plt.tight_layout()
    plt.savefig(experiment_plot_directory + "/{}_heatmap_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    # plt.show()
    plt.close()


def save_ant_histogram_data():
    """
    Save Ant histogram data.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]

    format: .npy and .pkl

    Note: pkl file has joint labels.
    """

    np.save(experiment_data_directory + "/{}_histogram_data_{}.npy".format(experiment_name, num_seeds), ant_histogram_data)

    df = pd.DataFrame(np.array(ant_histogram_data).T, columns=["hip_1", "ankle_1",  "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4" ,"ankle_4"])
    df.to_pickle(experiment_data_directory + "/{}_histogram_data_{}.pkl".format(experiment_name, num_seeds))


def save_ant_joint_angles(d):
    """
    Save Ant visited joint angles.

    histogram data:
    [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]

    @param d: PyMjData object (https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data)
        mujoco-py simulation data
    """

    ant_histogram_data[0].append(d.get_joint_qpos("hip_1"))
    ant_histogram_data[1].append(d.get_joint_qpos("ankle_1"))
    ant_histogram_data[2].append(d.get_joint_qpos("hip_2"))
    ant_histogram_data[3].append(d.get_joint_qpos("ankle_2"))
    ant_histogram_data[4].append(d.get_joint_qpos("hip_3"))
    ant_histogram_data[5].append(d.get_joint_qpos("ankle_3"))
    ant_histogram_data[6].append(d.get_joint_qpos("hip_4"))
    ant_histogram_data[7].append(d.get_joint_qpos("ankle_4"))


def collect_ant_data():

    pbar = tqdm(total=num_seeds)

    seeds = os.listdir(file)
    seeds = sorted([int(s[4:]) for s in seeds])

    for seed in seeds:
        pbar.set_description("Processing seed {}".format(seed))

        sim = AntHistogram(seed)
        sim.run()
        sim.cleanup()

        pbar.update(1)

    save_ant_histogram_data()


def plot_ant_data():

    ranges = get_ant_xml_data()

    plot_ant_histograms(ranges)

    plot_ant_heatmap(ranges)


class AntHistogram:
    """
    Controller for creating a histogram of visited joint angles for FetchReach.
    """

    LINE = "--------------------------------------------------------------------------------"

    def __init__(self, seed):
        """
        @param seed: int
            experiment seed
        """

        self.load_data_dir = file + "/seed{}".format(seed)
        self.t = int(time_steps)

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
            save_ant_joint_angles(self.env.env.sim.data)

            terminal = False

            max_steps_this_episode = 1000
            while not terminal and ((max_steps_this_episode <= 0) or (self.rlg.num_ep_steps() < max_steps_this_episode)):
                _, state, terminal, _ = self.rlg.rl_step()
                save_ant_joint_angles(self.env.env.sim.data)


def get_fetchrach_xml_data():

    if "melco2" in os.uname()[1]:
        anaconda_path = "/opt/anaconda3"
    elif "melco" in os.uname()[1]:
        anaconda_path = "/local/melco2/sschoepp/anaconda3"
    else:
        anaconda_path = os.getenv("HOME") + "/anaconda3"

    # model_xml = None
    # if env_name == "FetchReach-v1":
    #     model_xml = anaconda_path + "/envs/openai3.7/lib/python3.7/site-packages/gym/envs/robotics/assets/fetch/reach.xml"
    # if "v0" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"
    # elif "v1" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/reach.xml"
    # elif "v2" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/reach.xml"
    # elif "v3" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/reach.xml"
    # elif "v4" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenShoulderLiftSensor/assets/fetch/reach.xml"
    # elif "v5" in env_name:
    #     model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v5_BrokenJointsTBD/assets/fetch/reach.xml"4
    model_xml = anaconda_path + "/envs/openai3.7/lib/python3.7/site-packages/gym/envs/robotics/assets/fetch/reach.xml"

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

    fetchreach_histogram_data = np.load(experiment_data_directory + "/{}_histogram_data_{}.npy".format(experiment_name, num_seeds))

    fetchreach_histogram_data_clipped = []

    # clip data
    for i in range(len(ranges)):
        min = ranges[i][0]
        max = ranges[i][1]
        data = fetchreach_histogram_data[i]

        clipped = np.clip(data, min, max)

        fetchreach_histogram_data_clipped.append(clipped)

    def title():
        if suffix == "":
            title = "{}\n{}\n{}".format(algorithm, name, joint_name)
        else:
            title = "{} ({})\n{}\n{}".format(algorithm, suffix_eval, name, joint_name)
        return title

    index = 0  # shoulder_pan_joint
    joint_name = "shoulder pan joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.05, 0.05), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_shoulder_pan_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 1  # shoulder_lift_joint
    joint_name = "shoulder lift joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.05, 0.05), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_shoulder_lift_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 2  # upperarm_roll_joint
    joint_name = "upperarm roll joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.1, 0.1), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_upperarm_roll_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 3  # elbow_flex_joint
    joint_name = "elbow flex joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.1, 0.1), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_elbow_flex_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 4  # forearm_roll_joint
    joint_name = "forearm roll joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.1, 0.1), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_forearm_roll_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 5  # wrist_flex_joint
    joint_name = "wrist flex joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.05, 0.05), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_wrist_flex_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    plt.close()

    index = 6  # wrist_roll_joint
    joint_name = "wrist roll joint"
    df = pd.DataFrame(fetchreach_histogram_data_clipped[index], columns=["radians"])
    plot = sns.histplot(data=df, x="radians", color="tab:blue", stat="probability", bins=np.arange(ranges[index][0], ranges[index][1] + 0.1, 0.1), element="step")
    plot.set_title(title())
    plt.ylim(0, 1)
    plt.ylabel("probability")
    plt.savefig(experiment_plot_directory + "/{}_wrist_roll_joint_{}.jpg".format(experiment_name, num_seeds), dpi=300)
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

    fetchreach_histogram_data = np.load(experiment_data_directory + "/{}_histogram_data_{}.npy".format(experiment_name, num_seeds))

    fetchreach_histogram_data_normalized = []

    # normalize and clip data
    for i in range(len(ranges)):

        min = ranges[i][0]
        max = ranges[i][1]
        data = fetchreach_histogram_data[i]

        normalized = (data - min) / (max - min)

        normalized = np.clip(normalized, 0, 1)

        fetchreach_histogram_data_normalized.append(normalized)

    fetchreach_histogram_count_data = []
    bins = np.arange(0.0, 1.05, 0.025)

    for i in range(7):
        counts, _ = np.histogram(fetchreach_histogram_data_normalized[i], bins=bins)
        counts = counts / np.sum(counts)  # probability
        fetchreach_histogram_count_data.append(counts)

    df = pd.DataFrame(np.array(fetchreach_histogram_count_data).T, index=np.flip(bins[:-1]), columns=["shoulder pan", "shoulder lift", "upperarm roll", "elbow flex", "forearm roll", "wrist flex", "wrist roll"])

    y_labels = ["1.0", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "0.5", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "0.0"]
    heatmap = sns.heatmap(data=df[::-1], cmap=cmap, yticklabels=y_labels)

    algorithm_ = None

    if algorithm == "PPO":
        algorithm_ = "Proximal Policy Optimization (PPO)"
    elif algorithm == "SAC":
        algorithm_ = "Soft Actor-Critic (SAC)"

    if suffix == "":
        heatmap.set_title("{}".format(algorithm_))
    else:
        heatmap.set_title("{} ({})\n{}".format(algorithm, suffix_eval, name))
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.xlabel("joint")
    plt.ylabel("normalized angle")
    plt.tight_layout()
    plt.savefig(experiment_plot_directory + "/{}_heatmap_{}.jpg".format(experiment_name, num_seeds), dpi=300)
    # plt.show()
    plt.close()


def save_fetchreach_histogram_data():
    """
    Save FetchReach histogram data.

    histogram data:
    [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]

    format: .npy and .pkl

    Note: pkl file has joint labels.
    """

    np.save(experiment_data_directory + "/{}_histogram_data_{}.npy".format(experiment_name, num_seeds), fetchreach_histogram_data)

    df = pd.DataFrame(np.array(fetchreach_histogram_data).T, columns=["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"])
    df.to_pickle(experiment_data_directory + "/{}_histogram_data_{}.pkl".format(experiment_name, num_seeds))


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

    pbar = tqdm(total=num_seeds)

    seeds = os.listdir(file)
    seeds = sorted([int(s[4:]) for s in seeds])

    for seed in seeds:
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

        self.load_data_dir = file + "/seed{}".format(seed)
        self.t = int(time_steps)

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


def draw_histogram():
    global algorithm, env_name, time_steps, abnormal, crb, cm, rn, ant_histogram_data, fetchreach_histogram_data, suffix, suffix_eval, experiment_data_directory, experiment_plot_directory, experiment_name, name

    algorithm = None
    env_name = None
    time_steps = None

    abnormal = False
    crb = None
    cm = None
    rn = None

    ant_histogram_data = None
    fetchreach_histogram_data = None

    suffix = None
    suffix_eval = None
    experiment_data_directory = None
    experiment_plot_directory = None
    experiment_name = None

    name = None

    params = file.split("/")[-1].split("_")
    algorithm = params[0][0:3]
    env_info = params[1].split(":")
    env_name = env_info[0]
    time_steps = int(env_info[1])

    if "v0" != env_name[-2:] and "Ant-v2" != env_name and "FetchReach-v1" != env_name:
        abnormal = True
        time_steps *= 2
        if algorithm == "PPO":
            for p in params[2:]:
                if p.startswith("cm:"):
                    cm = eval(p.split(":")[1])
                elif p.startswith("rn:"):
                    rn = eval(p.split(":")[1])
            if cm and rn:
                suffix_eval = "retain no data"
            elif cm and not rn:
                suffix_eval = "retain network parameters"
            elif not cm and rn:
                suffix_eval = "retain memory"
            else:
                suffix_eval = "retain all data"
        if algorithm == "SAC":
            for p in params[2:]:
                if p.startswith("crb:"):
                    crb = eval(p.split(":")[1])
                elif p.startswith("rn:"):
                    rn = eval(p.split(":")[1])
            if crb and rn:
                suffix_eval = "retain no data"
            elif crb and not rn:
                suffix_eval = "retain network parameters"
            elif not crb and rn:
                suffix_eval = "retain replay buffer"
            else:
                suffix_eval = "retain all data"

    if "Ant" in env_name:
        env_folder_name = "ant"
    else:
        env_folder_name = "fetchreach"
        assert "FetchReach" in env_name, "draw_histogram: env_name does not contain Ant or FetchReach"

    if not abnormal:
        suffix = ""
        experiment_data_directory = os.path.join(os.getcwd(), "data", env_folder_name, env_name, algorithm)
        experiment_plot_directory = os.path.join(os.getcwd(), "plots", env_folder_name, env_name, algorithm)
    else:
        if algorithm == "PPO":
            suffix = "cm:{}_rn:{}".format(cm, rn)
        else:
            suffix = "crb:{}_rn:{}".format(crb, rn)
        experiment_data_directory = os.path.join(os.getcwd(), "data", env_folder_name, env_name, algorithm, suffix)
        experiment_plot_directory = os.path.join(os.getcwd(), "plots", env_folder_name, env_name, algorithm, suffix)
    os.makedirs(experiment_data_directory, exist_ok=True)
    os.makedirs(experiment_plot_directory, exist_ok=True)
    experiment_name = env_name + "_" + algorithm + (("_" + suffix) if suffix != "" else suffix)

    if "Ant" in env_name:

        # histogram data:
        # [hip_1, ankle_1,  hip_2, ankle_2, hip_3, ankle_3, hip_4 ,ankle_4]
        ant_histogram_data = [[], [], [], [], [], [], [], []]

        # name for plotting
        name = ""
        if env_name == "Ant-v2" or env_name == "AntEnv-v0":
            name = "No Fault"
        elif env_name == "AntEnv-v1":
            name = "Broken, Severed Limb"
        elif env_name == "AntEnv-v2":
            name = "Hip 4 ROM Restriction"
        elif env_name == "AntEnv-v3":
            name = "Ankle 4 ROM Restriction"
        elif env_name == "AntEnv-v4":
            name = "Broken, Unsevered Limb"

        if collect_data:
            collect_ant_data()

        plot_ant_data()

    elif "FetchReach" in env_name:

        # histogram data:
        # [shoulder_pan_joint, shoulder_lift_joint, upperarm_roll_joint, elbow_flex_joint, forearm_roll_joint, wrist_flex_joint, wrist_roll_joint]
        fetchreach_histogram_data = [[], [], [], [], [], [], []]

        # name for plotting
        name = ""
        if env_name == "FetchReachEnv-v0":
            name = "No Fault"
        elif env_name == "FetchReachEnvGE-v0":
            name = "No Fault (with Goal Elimination)"
        elif env_name == "FetchReachEnv-v1":
            name = "Shoulder Lift Reduced ROM Fault"
        elif env_name == "FetchReachEnvGE-v1":
            name = "Shoulder Lift Reduced ROM Fault (with Goal Elimination)"
        elif env_name == "FetchReachEnv-v4":
            name = "Shoulder Lift Sensor Fault"

        if collect_data:
            collect_fetchreach_data()

        plot_fetchreach_data()

    else:

        print("file argument does not include Ant or FetchReach")


if __name__ == "__main__":

    algorithm = None
    env_name = None
    time_steps = None

    abnormal = False
    crb = None
    cm = None
    rn = None

    ant_histogram_data = None
    fetchreach_histogram_data = None

    suffix = None
    suffix_eval = None
    experiment_data_directory = None
    experiment_plot_directory = None
    experiment_name = None

    name = None

    num_seeds = 10
    print(colored("the number of seeds is set to {}".format(num_seeds), "blue"))

    collect_data = args.collect_data

    """ant normal"""

    # PPO

    file = "/media/sschoepp/easystore/shared/ant/normal/PPOv2_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_d:cpu_ps:True_pss:33_resumed"
    draw_histogram()

    # SAC

    file = "/media/sschoepp/easystore/shared/ant/normal/SACv2_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_a:True_d:cuda_ps:True_pss:61_resumed"
    draw_histogram()

    """ant faulty"""

    # PPO v1

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v1/PPOv2_AntEnv-v1:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v1/PPOv2_AntEnv-v1:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:True_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v1/PPOv2_AntEnv-v1:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v1/PPOv2_AntEnv-v1:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:True_d:cpu_resumed"
    draw_histogram()

    # PPO v2

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v2/PPOv2_AntEnv-v2:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v2/PPOv2_AntEnv-v2:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:True_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v2/PPOv2_AntEnv-v2:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v2/PPOv2_AntEnv-v2:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:True_d:cpu_resumed"
    draw_histogram()

    # PPO v3

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v3/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v3/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:True_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v3/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v3/PPOv2_AntEnv-v3:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:True_d:cpu_resumed"
    draw_histogram()

    # PPO v4

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v4/PPOv2_AntEnv-v4:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v4/PPOv2_AntEnv-v4:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:False_rn:True_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v4/PPOv2_AntEnv-v4:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:False_d:cpu_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/ppo/v4/PPOv2_AntEnv-v4:600000000_Ant-v2:600000000_lr:0.000123_lrd:True_slrd:0.25_g:0.9839_ns:2471_mbs:1024_epo:5_eps:0.3_c1:1.0_c2:0.0019_cvl:False_mgn:0.5_gae:True_lam:0.911_hd:64_lstd:0.0_tef:3000000_ee:10_tmsf:50000000_cm:True_rn:True_d:cpu_resumed"
    draw_histogram()

    # SAC v1

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v1/SACv2_AntEnv-v1:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    # SAC v2

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v2/SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v2/SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v2/SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v2/SACv2_AntEnv-v2:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    # SAC v3

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v3/SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v3/SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v3/SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v3/SACv2_AntEnv-v3:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    # SAC v4

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v4/SACv2_AntEnv-v4:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v4/SACv2_AntEnv-v4:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:False_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v4/SACv2_AntEnv-v4:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:False_a:True_d:cuda_resumed"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/ant/faulty/sac/v4/SACv2_AntEnv-v4:20000000_Ant-v2:20000000_g:0.9646_t:0.0877_a:0.2_lr:0.001092_hd:256_rbs:500000_bs:512_mups:1_tui:1_tef:100000_ee:10_tmsf:1000000_crb:True_rn:True_a:True_d:cuda_resumed"
    draw_histogram()

    """fetchreach normal"""

    # PPO v0

    file = "/media/sschoepp/easystore/shared/fetchreach/normal/PPOv2_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_d:cpu_ps:True_pss:43"
    draw_histogram()

    # SAC v0

    file = "/media/sschoepp/easystore/shared/fetchreach/normal/SACv2_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_a:True_d:cuda_ps:True_pss:21"
    draw_histogram()

    """fetchreach faulty"""

    # PPO v1

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v1/PPOv2_FetchReachEnv-v1:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:False_rn:False_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v1/PPOv2_FetchReachEnv-v1:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:False_rn:True_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v1/PPOv2_FetchReachEnv-v1:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:True_rn:False_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v1/PPOv2_FetchReachEnv-v1:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:True_rn:True_d:cpu"
    draw_histogram()

    # PPO v4

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v4/PPOv2_FetchReachEnv-v4:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:False_rn:False_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v4/PPOv2_FetchReachEnv-v4:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:False_rn:True_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v4/PPOv2_FetchReachEnv-v4:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:True_rn:False_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v4/PPOv2_FetchReachEnv-v4:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:True_rn:True_d:cpu"
    draw_histogram()

    # PPO v6

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v6/PPOv2_FetchReachEnv-v6:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:False_rn:False_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v6/PPOv2_FetchReachEnv-v6:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:False_rn:True_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v6/PPOv2_FetchReachEnv-v6:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:True_rn:False_d:cpu"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/ppo/v6/PPOv2_FetchReachEnv-v6:6000000_FetchReachEnv-v0:6000000_lr:0.000275_lrd:True_g:0.848_ns:3424_mbs:8_epo:24_eps:0.3_c1:1.0_c2:0.0007_cvl:False_mgn:0.5_gae:True_lam:0.9327_hd:64_lstd:0.0_tef:30000_ee:10_tmsf:60000_cm:True_rn:True_d:cpu"
    draw_histogram()

    # SAC v1

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v1/SACv2_FetchReachEnv-v1:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v1/SACv2_FetchReachEnv-v1:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:True_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v1/SACv2_FetchReachEnv-v1:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:True_rn:False_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v1/SACv2_FetchReachEnv-v1:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:True_rn:True_a:True_d:cuda"
    draw_histogram()

    # SAC v4

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v4/SACv2_FetchReachEnv-v4:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v4/SACv2_FetchReachEnv-v4:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:True_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v4/SACv2_FetchReachEnv-v4:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:True_rn:False_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v4/SACv2_FetchReachEnv-v4:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:True_rn:True_a:True_d:cuda"
    draw_histogram()

    # SAC v6

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v6/SACv2_FetchReachEnv-v6:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:False_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v6/SACv2_FetchReachEnv-v6:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:False_rn:True_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v6/SACv2_FetchReachEnv-v6:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:True_rn:False_a:True_d:cuda"
    draw_histogram()

    file = "/media/sschoepp/easystore/shared/fetchreach/faulty/sac/v6/SACv2_FetchReachEnv-v6:2000000_FetchReachEnv-v0:2000000_g:0.8097_t:0.0721_a:0.2_lr:0.001738_hd:256_rbs:10000_bs:512_mups:1_tui:1_tef:10000_ee:10_tmsf:20000_crb:True_rn:True_a:True_d:cuda"
    draw_histogram()
