import os
import xml.etree.ElementTree as ET
from pathlib import Path
import gym
import custom_gym_envs
import numpy as np
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.utils import inverse_kinematics as ik


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
            self.model_xml = anaconda_path + "/envs/openai/lib/python3.9/site-packages/gym/envs/robotics/assets/fetch/reach.xml"
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
            elif name == "robot0:elbow_flex_joint":
                self.elbow_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:wrist_flex_joint":
                self.wrist_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:r_gripper_finger_joint":
                self.r_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
            elif name == "robot0:l_gripper_finger_joint":
                self.l_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)

        self.upperarm_roll_joint_range = np.array([-np.pi, np.pi])
        self.forearm_roll_joint_range = np.array([-np.pi, np.pi])
        self.wrist_roll_joint_range = np.array([-np.pi, np.pi])

        self.min_z = 0.42  # z less than this will result in a goal being in the table

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

        if goal[2] < self.min_z:

            print("goal {} in table".format(str(goal)))

        else:

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
            # N/A
            # 1: robot0:slide1
            # N/A
            # 2: robot0:slide2
            # N/A

            # 3: robot0:torso_lift_joint
            torso_lift_joint_pos = result.qpos[3]

            # 4: robot0:head_pan_joint
            # N/A
            # 5: robot0:head_tilt_joint
            # N/A

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
            # N/A
            # 14: robot0:l_gripper_finger_joint
            # N/A

            if result.success:

                # not included: upperarm_roll_joint, forearm_roll_joint, wrist_roll_joint (these have full ROM)
                torso_lift_joint_inrange = self.torso_lift_joint_range[0] < torso_lift_joint_pos < self.torso_lift_joint_range[1]
                shoulder_pan_joint_inrange = self.shoulder_pan_joint_range[0] < shoulder_pan_joint_pos < self.shoulder_pan_joint_range[1]
                shoulder_lift_joint_inrange = self.shoulder_lift_joint_range[0] < shoulder_lift_joint_pos < self.shoulder_lift_joint_range[1]
                elbow_flex_joint_inrange = self.elbow_flex_joint_range[0] < elbow_flex_joint_pos < self.elbow_flex_joint_range[1]
                wrist_flex_joint_inrange = self.wrist_flex_joint_range[0] < wrist_flex_joint_pos < self.wrist_flex_joint_range[1]

                reachable = torso_lift_joint_inrange and shoulder_pan_joint_inrange and shoulder_lift_joint_inrange and elbow_flex_joint_inrange and wrist_flex_joint_inrange

                if reachable:

                    qpos = {
                        "robot0:torso_lift_joint": torso_lift_joint_pos,
                        "robot0:shoulder_pan_joint": shoulder_pan_joint_pos,
                        "robot0:shoulder_lift_joint": shoulder_lift_joint_pos,
                        "robot0:upperarm_roll_joint": upperarm_roll_joint_pos,
                        "robot0:elbow_flex_joint": elbow_flex_joint_pos,
                        "robot0:forearm_roll_joint": forearm_roll_joint_pos,
                        "robot0:wrist_flex_joint": wrist_flex_joint_pos,
                        "robot0:wrist_roll_joint": wrist_roll_joint_pos,
                    }

                else:

                    print("goal {} not reachable".format(str(goal)))

                    print("torso_lift_joint:", self.torso_lift_joint_range, torso_lift_joint_pos, "inrange:", torso_lift_joint_inrange)
                    print("shoulder_pan_joint:", self.shoulder_pan_joint_range, shoulder_pan_joint_pos, "inrange:", shoulder_pan_joint_inrange)
                    print("shoulder_lift_joint:", self.shoulder_lift_joint_range, shoulder_lift_joint_pos, "inrange:", shoulder_lift_joint_inrange)
                    print("upperarm_roll_joint:", self.upperarm_roll_joint_range, upperarm_roll_joint_pos, "inrange: N/A")
                    print("elbow_flex_joint:", self.elbow_flex_joint_range, elbow_flex_joint_pos, "inrange:", elbow_flex_joint_inrange)
                    print("forearm_roll_joint:", self.forearm_roll_joint_range, forearm_roll_joint_pos, "inrange: N/A")
                    print("wrist_flex_joint:", self.wrist_flex_joint_range, wrist_flex_joint_pos, "inrange:", wrist_flex_joint_inrange)
                    print("wrist_roll_joint:", self.wrist_roll_joint_range, wrist_roll_joint_pos, "inrange: N/A")

                    print(self.line)

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

        np.random.seed(0)

        env = gym.make(self.env_name)
        env.reset()

        for i in range(20):

            initial_gripper_pos = np.array([1.34183226, 0.74910038, 0.53472284])
            target_range = 0.15

            goal = initial_gripper_pos + np.random.uniform(-target_range, target_range, size=3)

            qpos, reachable = self.check_reachable(goal)

            if reachable:
                for name, value in qpos.items():
                    env.sim.data.set_joint_qpos(name, value)
                for i in range(10):
                    env.sim.step()
                print(goal, "goal")
                print(env.sim.data.get_site_xpos('robot0:grip'), "xpos after setting joint angles and simulation steps")

            print(self.line)

    def test_taskspace(self):

        print(self.line)

        env = gym.make(self.env_name)
        env.reset()

        reachable_points = []
        unreachable_points = []

        bins = 10
        accuracy = 0.005  # radians

        initial_gripper_pos = np.array([1.34183226, 0.74910038, 0.53472284])
        target_range = 0.15

        for x in np.linspace(initial_gripper_pos[0] - target_range, initial_gripper_pos[0] + target_range, bins):
            for y in np.linspace(initial_gripper_pos[1] - target_range, initial_gripper_pos[1] + target_range, bins):
                for z in np.linspace(max(initial_gripper_pos[2] - target_range, self.min_z), initial_gripper_pos[2] + target_range, bins):
                    point = [x, y, z]
                    _, reachable = self.check_reachable(point)
                    if reachable:
                        reachable_points.append(point)
                    else:
                        unreachable_points.append(point)

                        gripper_target = np.array(point)
                        gripper_rotation = np.array([1., 0., 1., 0.])
                        env.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                        env.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
                        for _ in range(20):
                            env.sim.step()
                        xpos = env.sim.data.get_site_xpos('robot0:grip')
                        print(xpos, "xpos after setting gripper position/orientation and simulation steps")
                        print(self.line)

        print("num reachable:", len(reachable_points), "/", bins**3)
        print("num unreachable:", len(unreachable_points), "/", bins**3)

        np.save("{}_reachable_points_.npy".format(self.env_name), reachable_points)


if __name__ == "__main__":

    # env_name = "FetchReach-v1"
    env_name = "FetchReachEnv-v2"

    k = Kinematics(env_name)
    # k.check_reachable([1.34183265, 0.74910039, 0.53472272])  # starting position in FetchReach-v1
    # k.check_reachable([100, 100, 100])  # not reachable (impossible)
    # k.check_reachable([0, 0, 0.39])  # in table
    # k.test_accuracy()
    k.test_taskspace()
