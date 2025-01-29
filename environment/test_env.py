import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchReach-v3', max_episode_steps=100)