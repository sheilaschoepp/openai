import gymnasium as gym
import time

import custom_gym_envs

env = gym.make("AntEnv-F3", render_mode="human", camera_name="free")


for episode in range(1000):
    env.reset(seed=episode)
    for step in range(300):
        env.step(env.action_space.sample())
        time.sleep(0.05)

env.close()
