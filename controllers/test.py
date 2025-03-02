import gymnasium as gym
import gymnasium_robotics
import time

import custom_gym_envs

# env = gym.make("FetchReach-F1", render_mode="human", width=1600, height=800)
env = gym.make("FetchReach-F1")

for episode in range(1000):
    env.reset(seed=episode)
    for step in range(100):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        time.sleep(0.05)

env.close()
