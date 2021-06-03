"""
modifications:
set reward_type to dense
"""

import os
from gym import utils
from custom_gym_envs.envs.fetchpickandplace import FetchEnv_v1  # todo

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


class FetchPickAndPlaceEnvV1(FetchEnv_v1.FetchEnvV1, utils.EzPickle):  # todo
    def __init__(self, reward_type='dense'):  # modification here
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        FetchEnv_v1.FetchEnvV1.__init__(  # todo
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)