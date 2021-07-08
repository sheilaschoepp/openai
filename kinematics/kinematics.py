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

        model_xml = None
        # if self.env_name == "FetchReach-v1":
        #     model_xml = anaconda_path + "/envs/openai2/lib/python3.9/site-packages/gym/envs/robotics/assets/fetch/reach.xml"  # TODO fix this
        if self.env_name == "FetchReachEnv-v0":
            model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/kinematics.xml"
        elif self.env_name == "FetchReachEnv-v1":
            model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/kinematics.xml"
        elif self.env_name == "FetchReachEnv-v2":
            model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/kinematics.xml"
        elif self.env_name == "FetchReachEnv-v3":
            model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/kinematics.xml"
        elif self.env_name == "FetchReachEnv-v4":
            model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenGrip/assets/fetch/kinematics.xml"
        elif self.env_name == "FetchReachEnv-v999":
            model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_TEST/assets/fetch/kinematics.xml"

        robot_xml = model_xml[:-14] + "robot.xml"
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

        self.model_xml_string = assets.get_contents(model_xml)

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

        if goal[2] < 0.42:

            print("goal {} in table".format(str(goal)))

            return {}, False

        else:

            physics = mujoco.Physics.from_xml_string(self.model_xml_string)
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

            # print(result.success)  # todo add this into the equation somewhere

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

            reachable = (self.torso_lift_joint_range[0] < torso_lift_joint_pos < self.torso_lift_joint_range[1]) and \
                        (self.shoulder_pan_joint_range[0] < shoulder_pan_joint_pos < self.shoulder_pan_joint_range[1]) and \
                        (self.shoulder_lift_joint_range[0] < shoulder_lift_joint_pos < self.shoulder_lift_joint_range[1]) and \
                        (self.upperarm_roll_joint_range[0] < upperarm_roll_joint_pos < self.upperarm_roll_joint_range[1]) and \
                        (self.elbow_flex_joint_range[0] < elbow_flex_joint_pos < self.elbow_flex_joint_range[1]) and \
                        (self.forearm_roll_joint_range[0] < forearm_roll_joint_pos < self.forearm_roll_joint_range[1]) and \
                        (self.wrist_flex_joint_range[0] < wrist_flex_joint_pos < self.wrist_flex_joint_range[1]) and \
                        (self.wrist_roll_joint_range[0] < wrist_roll_joint_pos < self.wrist_roll_joint_range[1])

            # print("reachable:", reachable)
            # print("torso_lift_joint:", self.torso_lift_joint_range, torso_lift_joint_pos)
            # print("shoulder_pan_joint:", self.shoulder_pan_joint_range, shoulder_pan_joint_pos)
            # print("shoulder_lift_joint:", self.shoulder_lift_joint_range, shoulder_lift_joint_pos)
            # print("upperarm_roll_joint:", self.upperarm_roll_joint_range, upperarm_roll_joint_pos)
            # print("elbow_flex_joint:", self.elbow_flex_joint_range, elbow_flex_joint_pos)
            # print("forearm_roll_joint:", self.forearm_roll_joint_range, forearm_roll_joint_pos)
            # print("wrist_flex_joint:", self.wrist_flex_joint_range, wrist_flex_joint_pos)
            # print("wrist_roll_joint:", self.wrist_roll_joint_range, wrist_roll_joint_pos)

            if not reachable:

                qpos = {}

                print("goal {} not reachable".format(str(goal)))

            return qpos, reachable

    def test(self):

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
                print(env.sim.data.get_site_xpos('robot0:grip'), "gripper xpos after setting joint angles and stepping")

            print(self.line)


if __name__ == "__main__":

    env_name = "FetchReachEnv-v999"

    k = Kinematics(env_name)
    # k.check_reachable([1.34183265, 0.74910039, 0.53472272])  # starting position in FetchReach-v1
    # k.check_reachable([100, 100, 100])  # not reachable (impossible)
    # k.check_reachable([0, 0, 0.39])  # in table
    k.test()

