import gymnasium as gym
import time

import gymnasium_robotics

import custom_gym_envs

env = gym.make("FetchReach-F2", render_mode="human", width=1600, height=800)


for episode in range(1000):
    env.reset(seed=episode)
    for step in range(100):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        # print(obs["observation"])
        time.sleep(0.05)

env.close()
