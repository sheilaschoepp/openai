import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.utils import inverse_kinematics as ik

env_name = "FetchReachEnv-v999"

if "melco2" in os.uname()[1]:
    anaconda_path = "/opt/anaconda3"
elif "melco" in os.uname()[1]:
    anaconda_path = "/local/melco2/sschoepp/anaconda3"
else:
    anaconda_path = os.getenv("HOME") + "/anaconda3"

model_xml = None
# if env_name == "FetchReach-v1":
#     model_xml = anaconda_path + "/envs/openai2/lib/python3.9/site-packages/gym/envs/robotics/assets/fetch/reach.xml"  # TODO fix this
if env_name == "FetchReachEnv-v0":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/kinematics.xml"
elif env_name == "FetchReachEnv-v1":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/kinematics.xml"
elif env_name == "FetchReachEnv-v2":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/kinematics.xml"
elif env_name == "FetchReachEnv-v3":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/kinematics.xml"
elif env_name == "FetchReachEnv-v4":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenGrip/assets/fetch/kinematics.xml"
elif env_name == "FetchReachEnv-v999":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_TEST/assets/fetch/kinematics.xml"

robot_xml = model_xml[:-9] + "robot.xml"
tree = ET.parse(robot_xml)
root = tree.getroot()

torso_lift_joint_range = None
head_pan_joint_range = None
head_tilt_joint_range = None
shoulder_pan_joint_range = None
shoulder_lift_joint_range = None
elbow_flex_joint_range = None
wrist_flex_joint_range = None

for child in root.iter():
    attrib = child.attrib
    name = attrib.get("name")
    if name == "robot0:torso_lift_joint":
        torso_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    # elif name == "robot0:head_pan_joint":
    #     head_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    # elif name == "robot0:head_tilt_joint":
    #     head_tilt_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:shoulder_pan_joint":
        shoulder_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:shoulder_lift_joint":
        shoulder_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:elbow_flex_joint":
        elbow_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:wrist_flex_joint":
        wrist_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)

model_xml_string = assets.get_contents(model_xml)

physics = mujoco.Physics.from_xml_string(model_xml_string)
site_name = "robot0:grip"
target_pos = np.array([1, 2, 3])
target_quat = np.array([1., 0., 1., 0.])  # in fetch_env._set_action they always set the orientation to a static value
joint_names = None  # all joints will be manipulated to reach the target position and quaternion

result = ik.qpos_from_site_pose(physics=physics,
                                site_name=site_name,
                                target_pos=target_pos,
                                target_quat=target_quat,
                                joint_names=joint_names)
print(result.success, result.qpos)

# At runtime the positions and orientations of all joints defined in the model are stored in the vector mjData.qpos,
# in the order in which the appear in the kinematic tree (source: http://www.mujoco.org/book/XMLreference.html#joint)
# 0: robot0:slide0
# 1: robot0:slide1
# 2: robot0:slide2

# 3: robot0:torso_lift_joint

# 4: robot0:head_pan_joint
# 5: robot0:head_tilt_joint

# 6: robot0:shoulder_pan_joint
# 7: robot0:shoulder_lift_joint
# 8: robot0:upperarm_roll_joint
# 9: robot0:elbow_flex_joint
# 10: robot0:forearm_roll_joint
# 11: robot0:wrist_flex_joint
# 12: robot0:wrist_roll_joint

# 13: robot0:r_gripper_finger_joint
# 14: robot0:l_gripper_finger_joint

