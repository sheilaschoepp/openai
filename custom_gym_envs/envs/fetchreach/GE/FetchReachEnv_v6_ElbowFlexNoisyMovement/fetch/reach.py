"""
modifications:
change from in import of fetch_env
changed distance_threshold from 0.05 to 0.001
added goal_elimination kwarg
passed goal_elimination argument to FetchEnv class init
"""

import os
from gym import utils
from custom_gym_envs.envs.fetchreach.FetchReachEnv_v6_ElbowFlexNoisyMovement import fetch_env  # modification here

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', goal_elimination=False):  # modification here
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.001,  # modification here
            initial_qpos=initial_qpos, reward_type=reward_type, goal_elimination=goal_elimination)  # modification here
        utils.EzPickle.__init__(self)
