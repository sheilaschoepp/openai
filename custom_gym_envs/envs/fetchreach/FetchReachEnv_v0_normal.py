import os
from gym import utils
# from gym.envs.robotics import fetch_env  # todo: comment out import of fetch_env
from custom_gym_envs.envs.fetchreach import FetchEnv_v0_normal as fetch_env  # todo: add import of FetchEnv_v<#>_<description>.py file as fetch_env


# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')
MODEL_XML_PATH = "/Documents/openai/custom_gym_envs/xml/fetchreach/FetchReachEnv_v0_normal_reach.xml"  # todo: replace MODEL_XML_PATH


class FetchReachEnvV0(fetch_env.FetchEnvV0, utils.EzPickle):  # todo: (1) add version to FetchReachEnv, (2) add version to FetchEnv
    def __init__(self, reward_type='dense'):  # todo: change sparse to dense
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnvV0.__init__(  # todo: add version to FetchEnv
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)