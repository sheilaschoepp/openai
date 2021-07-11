import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import gym
import numpy as np
from dm_control import mujoco
from dm_control.utils import inverse_kinematics as ik

import custom_gym_envs


class Kinematics:

    def __init__(self, env_name):
        """
        Initialize Kinematics class by loading thx XML as a string.

        @param env_name: string
            name of MuJoCo Gym environment
        """

        self.env_name = env_name

        if "melco2" in os.uname()[1]:
            anaconda_path = "/opt/anaconda3"
        elif "melco" in os.uname()[1]:
            anaconda_path = "/local/melco2/sschoepp/anaconda3"
        else:
            anaconda_path = os.getenv("HOME") + "/anaconda3"

        self.model_xml = None
        if self.env_name == "FetchReach-v1":
            self.model_xml = anaconda_path + "/envs/openai2/lib/python3.7/site-packages/gym/envs/robotics/assets/fetch/reach.xml"
        if self.env_name == "FetchReachEnv-v0":
            self.model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"
        elif self.env_name == "FetchReachEnv-v1":
            self.model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/reach.xml"
        elif self.env_name == "FetchReachEnv-v2":
            self.model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/reach.xml"
        elif self.env_name == "FetchReachEnv-v3":
            self.model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/reach.xml"
        elif self.env_name == "FetchReachEnv-v4":
            self.model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenGrip/assets/fetch/reach.xml"
        elif self.env_name == "FetchReachEnv-v999":
            self.model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_TEST/assets/fetch/reach.xml"

        robot_xml = self.model_xml[:-9] + "robot.xml"
        tree = ET.parse(robot_xml)
        root = tree.getroot()

        for child in root.iter():
            attrib = child.attrib
            name = attrib.get("name")
            if name == "robot0:torso_lift_joint":
                self.torso_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:head_pan_joint":
                self.head_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:head_tilt_joint":
                self.head_tilt_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:shoulder_pan_joint":
                self.shoulder_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:shoulder_lift_joint":
                self.shoulder_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:upperarm_roll_joint":
                if attrib.get("range") is not None:
                    self.upperarm_roll_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
                else:
                    self.upperarm_roll_joint_range = np.array([-np.inf, np.inf])
            elif name == "robot0:elbow_flex_joint":
                self.elbow_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:forearm_roll_joint":
                if attrib.get("range") is not None:
                    self.forearm_roll_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
                else:
                    self.forearm_roll_joint_range = np.array([-np.inf, np.inf])
            elif name == "robot0:wrist_flex_joint":
                self.wrist_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:wrist_roll_joint":
                if attrib.get("range") is not None:
                    self.wrist_roll_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
                else:
                    self.wrist_roll_joint_range = np.array([-np.inf, np.inf])
            elif name == "robot0:r_gripper_finger_joint":
                self.r_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:l_gripper_finger_joint":
                self.l_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)

        self.line = "------------------------------------------------------------------------------------------------"

    def check_reachable(self, goal):
        """
        Check if a goal is reachable.

        Note: If a goal is in the table (z < 0.42), it is not reachable.  Otherwise, compute the joint angles required
        to reach the goal and verify that they fall within the allowed joint ranges.

        @param goal: float64 numpy array with shape (3,)
            the (target) goal position of the robot end effector

        @return reachable: bool
            if true, goal is not in the table and is reachable
        """

        qpos = {}
        reachable = False

        physics = mujoco.Physics.from_xml_path(self.model_xml)
        site_name = "robot0:grip"
        target_pos = np.array(goal)
        target_quat = np.array([1., 0., 1., 0.])  # in fetch_env._env_setup and fetch_env._set_action they always set the orientation to a static value
        joint_names = None  # all joints will be manipulated to reach the target position and quaternion

        result = ik.qpos_from_site_pose(physics=physics,
                                        site_name=site_name,
                                        target_pos=target_pos,
                                        target_quat=target_quat,
                                        joint_names=joint_names)

        # At runtime the positions and orientations of all joints defined in the model are stored in the vector mjData.qpos,
        # in the order in which the appear in the kinematic tree (source: http://www.mujoco.org/book/XMLreference.html#joint)
        # 0: robot0:slide0
        slide0_pos = result.qpos[0]
        # 1: robot0:slide1
        slide1_pos = result.qpos[1]
        # 2: robot0:slide2
        slide2_pos = result.qpos[2]

        # 3: robot0:torso_lift_joint
        torso_lift_joint_pos = result.qpos[3]

        # 4: robot0:head_pan_joint
        head_pan_joint_pos = result.qpos[4]
        # 5: robot0:head_tilt_joint
        head_tilt_joint_pos = result.qpos[5]

        # 6: robot0:shoulder_pan_joint
        shoulder_pan_joint_pos = result.qpos[6]
        # 7: robot0:shoulder_lift_joint
        shoulder_lift_joint_pos = result.qpos[7]
        # 8: robot0:upperarm_roll_joint
        upperarm_roll_joint_pos = result.qpos[8]
        # 9: robot0:elbow_flex_joint
        elbow_flex_joint_pos = result.qpos[9]
        # 10: robot0:forearm_roll_joint
        forearm_roll_joint_pos = result.qpos[10]
        # 11: robot0:wrist_flex_joint
        wrist_flex_joint_pos = result.qpos[11]
        # 12: robot0:wrist_roll_joint
        wrist_roll_joint_pos = result.qpos[12]

        # 13: robot0:r_gripper_finger_joint
        r_gripper_finger_joint_pos = result.qpos[13]
        # 14: robot0:l_gripper_finger_joint
        l_gripper_finger_joint_pos = result.qpos[14]

        if result.success:

            torso_lift_joint_inrange = self.torso_lift_joint_range[0] <= torso_lift_joint_pos <= self.torso_lift_joint_range[1]
            shoulder_pan_joint_inrange = self.shoulder_pan_joint_range[0] <= shoulder_pan_joint_pos <= self.shoulder_pan_joint_range[1]
            shoulder_lift_joint_inrange = self.shoulder_lift_joint_range[0] <= shoulder_lift_joint_pos <= self.shoulder_lift_joint_range[1]
            upperarm_roll_joint_inrange = self.upperarm_roll_joint_range[0] <= upperarm_roll_joint_pos <= self.upperarm_roll_joint_range[1]
            elbow_flex_joint_inrange = self.elbow_flex_joint_range[0] <= elbow_flex_joint_pos <= self.elbow_flex_joint_range[1]
            forearm_roll_joint_inrange = self.forearm_roll_joint_range[0] <= forearm_roll_joint_pos <= self.forearm_roll_joint_range[1]
            wrist_flex_joint_inrange = self.wrist_flex_joint_range[0] <= wrist_flex_joint_pos <= self.wrist_flex_joint_range[1]
            wrist_roll_joint_inrange = self.wrist_roll_joint_range[0] <= wrist_roll_joint_pos <= self.wrist_roll_joint_range[1]
            r_gripper_finger_joint_inrange = self.r_gripper_finger_joint_range[0] <= r_gripper_finger_joint_pos <= self.r_gripper_finger_joint_range[1]
            l_gripper_finger_joint_inrange = self.l_gripper_finger_joint_range[0] <= l_gripper_finger_joint_pos <= self.l_gripper_finger_joint_range[1]

            reachable = torso_lift_joint_inrange and \
                        shoulder_pan_joint_inrange and \
                        shoulder_lift_joint_inrange and \
                        upperarm_roll_joint_inrange and \
                        elbow_flex_joint_inrange and \
                        forearm_roll_joint_inrange and \
                        wrist_flex_joint_inrange and \
                        wrist_roll_joint_inrange and \
                        r_gripper_finger_joint_inrange and \
                        l_gripper_finger_joint_inrange

            qpos = {
                "robot0:slide0": slide0_pos,
                "robot0:slide1": slide1_pos,
                "robot0:slide2": slide2_pos,

                "robot0:torso_lift_joint": torso_lift_joint_pos,

                "robot0:head_pan_joint": head_pan_joint_pos,
                "robot0:head_tilt_joint": head_tilt_joint_pos,

                "robot0:shoulder_pan_joint": shoulder_pan_joint_pos,
                "robot0:shoulder_lift_joint": shoulder_lift_joint_pos,
                "robot0:upperarm_roll_joint": upperarm_roll_joint_pos,
                "robot0:elbow_flex_joint": elbow_flex_joint_pos,
                "robot0:forearm_roll_joint": forearm_roll_joint_pos,
                "robot0:wrist_flex_joint": wrist_flex_joint_pos,
                "robot0:wrist_roll_joint": wrist_roll_joint_pos,

                "robot0:r_gripper_finger_joint": r_gripper_finger_joint_pos,
                "robot0:l_gripper_finger_joint": l_gripper_finger_joint_pos,
            }

            # if not reachable:
            #
            #     print("{} not reachable".format(str(goal)))
            #
            #     if not torso_lift_joint_inrange:
            #         print("torso_lift_joint_range:", self.torso_lift_joint_range, "position:", torso_lift_joint_pos)
            #     if not shoulder_pan_joint_inrange:
            #         print("shoulder_pan_joint_range:", self.shoulder_pan_joint_range, "position:", shoulder_pan_joint_pos)
            #     if not shoulder_lift_joint_inrange:
            #         print("shoulder_lift_joint_range:", self.shoulder_lift_joint_range, "position:", shoulder_lift_joint_pos)
            #     if not upperarm_roll_joint_inrange:
            #         print("upperarm_roll_joint_range:", self.upperarm_roll_joint_range, "position:", upperarm_roll_joint_pos)
            #     if not elbow_flex_joint_inrange:
            #         print("elbow_flex_joint_range:", self.elbow_flex_joint_range, "position:", elbow_flex_joint_pos)
            #     if not forearm_roll_joint_inrange:
            #         print("forearm_roll_joint_range:", self.forearm_roll_joint_range, "position:", forearm_roll_joint_pos)
            #     if not wrist_flex_joint_inrange:
            #         print("wrist_flex_joint_range:", self.wrist_flex_joint_range, "position:", wrist_flex_joint_pos)
            #     if not wrist_roll_joint_inrange:
            #         print("wrist_roll_joint_range:", self.wrist_roll_joint_range, "position:", wrist_roll_joint_pos)
            #     if not r_gripper_finger_joint_inrange:
            #         print("r_gripper_finger_joint_range:", self.r_gripper_finger_joint_range, "position:", r_gripper_finger_joint_pos)
            #     if not l_gripper_finger_joint_inrange:
            #         print("l_tripper_finger_joint_range:", self.l_gripper_finger_joint_range, "position:", l_gripper_finger_joint_pos)

        return qpos, reachable

    def test_accuracy(self):
        """
        Test the accuracy of the kinematics.

        Note:
        - Sample goal.
        - Use inverse kinematics to obtain joint positions to reach goal.
        - Apply joint positions to robot and step to reach the positions.
        - Get the position of the robot's end effector.
        - Confirm sampled goal and robot's end effector position are similar.
        """

        print(self.line)

        env = gym.make(self.env_name)

        np.random.seed(0)
        env.seed(0)

        episodes = 100
        reachable_goals = 0  # number of goals classified as reachable
        successes = 0  # number of times the goal is reached

        for i in range(episodes):

            obs = env.reset()

            goal = obs["desired_goal"]

            qpos, reachable = self.check_reachable(goal)

            if reachable:

                reachable_goals += 1

                for name, value in qpos.items():
                    env.sim.data.set_joint_qpos(name, value)

                env.sim.forward()

                xpos = env.sim.data.get_site_xpos('robot0:grip')

                d = np.linalg.norm(xpos - goal, axis=-1)  # euclidean distance
                success = d <= env.distance_threshold

                if success:  # goal reached through simulation
                    successes += 1

        print("reachable goals:", reachable_goals)
        print("successes:", successes)

        print("accuracy:", str(int(successes/reachable_goals*100)) + "%")

        print(self.line)

    def test_task_space(self):
        """
        Test to see if goals within the goal space are reachable.

        Note: Use inverse kinematics to compute the required joint positions to reach a target position (goal).
        Check: If a goal is deemed unreachable, send a command to the robot to reach the unreachable goal.
        If working correctly, the gripper xpos will never reach target position (goal).
        """

        # print(self.line)

        env = gym.make(self.env_name)
        env.reset()

        reachable_goals = []
        unreachable_goals = []

        initial_gripper_pos = np.array([1.34183226, 0.74910038, 0.53472284])
        target_range = 0.15

        bins = 10  # number of x, y and z values to test from within the goal space

        num_false_negatives = 0

        for x in np.linspace(initial_gripper_pos[0] - target_range, initial_gripper_pos[0] + target_range, bins):
            for y in np.linspace(initial_gripper_pos[1] - target_range, initial_gripper_pos[1] + target_range, bins):
                for z in np.linspace(initial_gripper_pos[2] - target_range, initial_gripper_pos[2] + target_range, bins):

                    goal = [x, y, z]

                    _, reachable = self.check_reachable(goal)

                    if reachable:
                        reachable_goals.append(goal)
                    else:
                        unreachable_goals.append(goal)

                        # set target position (goal) and orientation of the gripper
                        gripper_target = np.array(goal)
                        gripper_rotation = np.array([1., 0., 1., 0.])
                        env.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                        env.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)

                        # perform simulation steps
                        for _ in range(20):

                            env.sim.step()

                        # todo test this and move it into loop
                        # get gripper position to compare to target postion (goal)
                        xpos = env.sim.data.get_site_xpos('robot0:grip')

                        # print(xpos, "xpos after setting gripper position/orientation and simulation steps")

                        d = np.linalg.norm(xpos - goal, axis=-1)  # euclidean distance
                        false_negative = d <= env.distance_threshold
                        # print("false negative:", false_negative)

                        if false_negative:
                            num_false_negatives += 1
                            # break  todo

                        # print(self.line)

        print("num reachable:", len(reachable_goals), "/", bins**3)
        print("num unreachable:", len(unreachable_goals), "/", bins**3)

        print("false negatives:", str(num_false_negatives), "/", len(unreachable_goals))

        # np.save("{}_reachable_points_{}.npy".format(self.env_name, bins), reachable_points)
        # np.save("{}_unreachable_points_{}.npy".format(self.env_name, bins), unreachable_points)


if __name__ == "__main__":

    # env_name = "FetchReach-v1"
    env_name = "FetchReachEnv-v999"

    k = Kinematics(env_name)
    # k.check_reachable([1.34183265, 0.74910039, 0.53472272])  # starting position in FetchReach-v1
    # k.check_reachable([100, 100, 100])  # not reachable (impossible)
    # k.check_reachable([0, 0, 0.39])  # in table
    k.test_accuracy()
    # k.test_task_space()
