import gymnasium as gym
import gymnasium_robotics

from gymnasium.wrappers import FlattenObservation

# gym.register_envs(gymnasium_robotics)

env = gym.make('FetchReach-v3')
obs, _ = env.reset(seed=0)
print(obs)

env = FlattenObservation(env)
obs, _ = env.reset(seed=0)
print(obs)