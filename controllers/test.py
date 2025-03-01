import gymnasium as gym
import time

import gymnasium_robotics

import custom_gym_envs

env = gym.make("FetchReach-F0", render_mode="human", width=1600, height=800)


for episode in range(1000):
    env.reset(seed=episode)
    for step in range(100):
        env.step(env.action_space.sample())
        time.sleep(0.05)

env.close()
