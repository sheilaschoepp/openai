import gym
import numpy as np
import os
import pickle


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

        self.env.sim.set_state(initial_state_copy)

        self.env.sim.forward()  # forward dynamics: same as mj_step but do not integrate in time
