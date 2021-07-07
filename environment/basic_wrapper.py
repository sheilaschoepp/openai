import gym
import numpy as np
import os
import pickle
import copy


class BasicWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.hostname = os.uname()[1]
        self.localhosts = ["melco", "Legion", "amii", "mehran"]
        self.computecanada = not any(host in self.hostname for host in self.localhosts)

        self.env = env

    def reset(self):
        if not self.computecanada:
            environment_path = os.getenv("HOME") + "/Documents/openai/environment"
        else:
            environment_path = os.getenv("HOME") + "/scratch/openai/environment"

        with open(environment_path + "/fetchreach_initial_state.pkl", "rb") as f:
            initial_state_copy = pickle.load(f)

        self.env.initial_state = copy.deepcopy(initial_state_copy)

        obs = self.env.reset()

        return obs