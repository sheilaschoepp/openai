import argparse
import time

import gym
import numpy as np

import custom_gym_envs

parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Arguments")

parser.add_argument("-e", "--env_name", default="FetchReachEnv-v4",
                    help="name of MuJoCo Gym environment (default: FetchReachEnvGE-v0)")

args = parser.parse_args()

env = gym.make(args.env_name)

# IMPORTANT: the following three lines do not work for v3 (it cannot find a goal that is reachable)
# env.kinematics.test_percent_reachable()
# env.kinematics.test_task_space()
env.kinematics.test_render_kinematics()

env.seed(0)
np.random.seed(0)

for e in range(100):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
        time.sleep(2)
