import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import copy

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

    def check_reachable(self, env_copy, goal):
        """
        Check if a goal is reachable.

        Note: If a goal is in the table (z < 0.42), it is not reachable.  Otherwise, compute the joint angles required
        to reach the goal and verify that they fall within the allowed joint ranges.

        @param env_copy: MuJoCo Gym environment copy
            a copy of the learning environment
        @param goal: float64 numpy array with shape (3,)
            the (target) goal position of the robot end effector

        @return reachable: bool
            if true, goal is not in the table and is reachable
        """

        reachable = False
        qpos = {}

        physics = mujoco.Physics.from_xml_path(self.model_xml)

        state = env_copy.env.sim.get_state()
        flattened_state = np.append(state.qpos, state.qvel)
        physics.set_state(flattened_state)

        site_name = "robot0:grip"
        target_pos = np.array(goal)
        target_quat = np.array([1., 0., 1., 0.])  # in fetch_env._env_setup and fetch_env._set_action they always set the orientation to a static value
        joint_names = ["robot0:shoulder_pan_joint",  # manipulate only these joints to reach the target position and quaternion
                       "robot0:shoulder_lift_joint",
                       "robot0:upperarm_roll_joint",
                       "robot0:elbow_flex_joint",
                       "robot0:forearm_roll_joint",
                       "robot0:wrist_flex_joint",
                       "robot0:wrist_roll_joint"]

        result = ik.qpos_from_site_pose(physics=physics,
                                        site_name=site_name,
                                        target_pos=target_pos,
                                        target_quat=target_quat,
                                        joint_names=joint_names,
                                        max_steps=100)

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

            shoulder_pan_joint_inrange = self.shoulder_pan_joint_range[0] <= shoulder_pan_joint_pos <= self.shoulder_pan_joint_range[1]
            shoulder_lift_joint_inrange = self.shoulder_lift_joint_range[0] <= shoulder_lift_joint_pos <= self.shoulder_lift_joint_range[1]
            upperarm_roll_joint_inrange = self.upperarm_roll_joint_range[0] <= upperarm_roll_joint_pos <= self.upperarm_roll_joint_range[1]
            elbow_flex_joint_inrange = self.elbow_flex_joint_range[0] <= elbow_flex_joint_pos <= self.elbow_flex_joint_range[1]
            forearm_roll_joint_inrange = self.forearm_roll_joint_range[0] <= forearm_roll_joint_pos <= self.forearm_roll_joint_range[1]
            wrist_flex_joint_inrange = self.wrist_flex_joint_range[0] <= wrist_flex_joint_pos <= self.wrist_flex_joint_range[1]
            wrist_roll_joint_inrange = self.wrist_roll_joint_range[0] <= wrist_roll_joint_pos <= self.wrist_roll_joint_range[1]

            reachable = shoulder_pan_joint_inrange and \
                        shoulder_lift_joint_inrange and \
                        upperarm_roll_joint_inrange and \
                        elbow_flex_joint_inrange and \
                        forearm_roll_joint_inrange and \
                        wrist_flex_joint_inrange and \
                        wrist_roll_joint_inrange

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

            if not reachable:

                print("{} not reachable".format(str(goal)))

                if not shoulder_pan_joint_inrange:
                    print("shoulder_pan_joint_range:", np.around(np.degrees(self.shoulder_pan_joint_range), 2), "position:", np.around(np.degrees(shoulder_pan_joint_pos), 2))
                if not shoulder_lift_joint_inrange:
                    print("shoulder_lift_joint_range:", np.around(np.degrees(self.shoulder_lift_joint_range), 2), "position:", np.around(np.degrees(shoulder_lift_joint_pos), 2))
                if not upperarm_roll_joint_inrange:
                    print("upperarm_roll_joint_range:", np.around(np.degrees(self.upperarm_roll_joint_range), 2), "position:", np.around(np.degrees(upperarm_roll_joint_pos), 2))
                if not elbow_flex_joint_inrange:
                    print("elbow_flex_joint_range:", np.around(np.degrees(self.elbow_flex_joint_range), 2), "position:", np.around(np.degrees(elbow_flex_joint_pos), 2))
                if not forearm_roll_joint_inrange:
                    print("forearm_roll_joint_range:", np.around(np.degrees(self.forearm_roll_joint_range), 2), "position:", np.around(np.degrees(forearm_roll_joint_pos), 2))
                if not wrist_flex_joint_inrange:
                    print("wrist_flex_joint_range:", np.around(np.degrees(self.wrist_flex_joint_range), 2), "position:", np.around(np.degrees(wrist_flex_joint_pos), 2))
                if not wrist_roll_joint_inrange:
                    print("wrist_roll_joint_range:", np.around(np.degrees(self.wrist_roll_joint_range), 2), "position:", np.around(np.degrees(wrist_roll_joint_pos), 2))

                print(self.line)

                pass

        return reachable, qpos, result

    def test_accuracy(self):
        """
        Test the accuracy of the kinematics.

        Note:
        - Sample goal.
        - Use inverse kinematics to obtain joint positions to reach goal.
        - Apply joint positions to robot and forward to reach the positions.
        - Get the position of the robot's end effector.
        - Confirm sampled goal and robot's end effector position are similar.
        """

        print(self.line)

        env = gym.make(self.env_name)
        env_copy = copy.deepcopy(env)  # we use a copy of the environment with the check_reachable method

        # np.random.seed(0)
        # env.seed(0)

        episodes = 100

        result_successes = 0

        kin_reachable_goals = 0
        kin_unreachable_goals = 0

        for i in range(episodes):

            obs = env_copy.reset()
            goal = obs["desired_goal"]

            reachable, qpos, result = self.check_reachable(env_copy, goal)

            if reachable:

                kin_reachable_goals += 1

            else:

                kin_unreachable_goals += 1

            if result.success:

                result_successes += 1

            #     for name, value in qpos.items():  todo: remove this because it doesn't help us, we are setting joint positions to invalid values
            #         env_copy.sim.data.set_joint_qpos(name, value)
            #
            #     env_copy.sim.forward()
            #
            #     xpos = env_copy.sim.data.get_site_xpos('robot0:grip')
            #
            #     d = np.linalg.norm(xpos - goal, axis=-1)  # euclidean distance
            #     goal_reached = d <= env_copy.distance_threshold
            #
            #     if not reachable and goal_reached:
            #         kin_inaccurate_goals += 1

        print("kinematics result rate:", str(result_successes / episodes * 100) + "%")

        print("kin reachable goals:", str(int(kin_reachable_goals / episodes * 100)) + "%")
        print("kin unreachable goals:", str(int(kin_unreachable_goals / episodes * 100)) + "%")

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

                    reachable, _, _ = self.check_reachable(goal)

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
                            # break todo

        print("num reachable:", len(reachable_goals), "/", bins**3)
        print("num unreachable:", len(unreachable_goals), "/", bins**3)

        print("false negatives:", str(num_false_negatives), "/", len(unreachable_goals))

        # np.save("{}_reachable_points_{}.npy".format(self.env_name, bins), reachable_points)
        # np.save("{}_unreachable_points_{}.npy".format(self.env_name, bins), unreachable_points)

    def test_unreachable(self):
        """
        Test an unreachable point.

        Note:
        - Create unreachable goal.
        - Use inverse kinematics to obtain joint positions to reach goal.
        - Apply joint positions to robot and forward to reach the positions.
        - Get the position of the robot's end effector.
        - Get the position of the robot's joints.
        """

        print(self.line)

        env = gym.make(self.env_name)

        np.random.seed(0)
        env.seed(0)

        for _ in range(10):

            obs = env.reset()

            # goal = np.array([1.0, 1.0, 1.0])
            # goal = np.array([0.7, 0.7, 0.7])
            goal = obs["desired_goal"]
            print("goal:", goal)
            print(self.line)

            reachable, qpos, result = self.check_reachable(copy.deepcopy(env), goal)

            if qpos is not {}:
                for name, value in qpos.items():
                    print(name, value)
                print(self.line)

            for name, value in qpos.items():
                env.sim.data.set_joint_qpos(name, value)

            env.sim.forward()

            env.render()
            time.sleep(2)

            if qpos is not {}:
                for name, _ in qpos.items():
                    print(name, env.sim.data.get_joint_qpos(name))
                print(self.line)

            xpos = env.sim.data.get_site_xpos('robot0:grip')

            d = np.linalg.norm(xpos - goal, axis=-1)  # euclidean distance
            success = d <= env.distance_threshold

            print("inverse kinematics results:")
            print("success:", result.success)
            print("err_norm:", result.err_norm)  # must be < 1e-14 to have success
            print("steps:", result.steps)  # must be < 100 to have success
            if result.success:
                print(self.line)
                print("forward results:")
                print("success (reached goal):", success)
                print("distance:", d)
                print(self.line)
                print("check reachable results:")
                print("reachable:", reachable)  # joint pos must be within strict limits

            print(self.line)


if __name__ == "__main__":

    env_name = "FetchReachEnv-v0"
    # env_name = "FetchReachEnv-v999"

    k = Kinematics(env_name)
    # k.check_reachable([1.34183265, 0.74910039, 0.53472272])  # starting position in FetchReach-v1
    # k.check_reachable([100, 100, 100])  # not reachable (impossible)
    # k.check_reachable([0, 0, 0.39])  # in table
    k.test_accuracy()
    # k.test_task_space()
    # k.test_unreachable()