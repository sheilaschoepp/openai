import gym
import numpy as np


class FetchReachObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        """Observation is a dictionary.  Create a new observation space using space of 'observation' and 'desired_goal' dictionary entries."""

        super(FetchReachObservationWrapper, self).__init__(env)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

    def observation(self, observation):
        """Observation is a dictionary.  Create a new numpy array using 'observation' and 'desired_goal' dictionary entries."""

        new_observation = np.concatenate((observation["observation"], observation["desired_goal"]))

        return new_observation
