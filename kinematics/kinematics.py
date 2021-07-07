import os
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
if env_name == "FetchReach-v1":
    model_xml = anaconda_path + "/envs/openai2/lib/python3.9/site-packages/gym/envs/robotics/assets/fetch/full.xml"
elif env_name == "FetchReachEnv-v0":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/full.xml"
elif env_name == "FetchReachEnv-v1":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/full.xml"
elif env_name == "FetchReachEnv-v2":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/full.xml"
elif env_name == "FetchReachEnv-v3":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/full.xml"
elif env_name == "FetchReachEnv-v4":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenGrip/assets/fetch/full.xml"
elif env_name == "FetchReachEnv-v999":
    model_xml = str(Path.home()) + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_TEST/assets/fetch/full.xml"

model_xml_string = assets.get_contents(model_xml)

physics = mujoco.Physics.from_xml_string(model_xml_string)
site_name = "robot0:grip"
target_pos = np.array([1, 2, 3])
target_quat = np.array([1., 0., 1., 0.])  # in fetch_env._set_action they always set the orientation to a static value
joint_names = None  # all joints will be manipulated to reach the target position and quaternion

result = ik.qpos_from_site_pose(physics=physics, site_name=site_name, target_pos=target_pos, target_quat=target_quat, joint_names=joint_names)
print(result.success, result.qpos)