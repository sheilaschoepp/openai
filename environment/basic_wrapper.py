import gym
import numpy as np
import os
import pickle
import copy


class BasicWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.env = env