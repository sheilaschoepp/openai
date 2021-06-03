import custom_gym_envs
import gym
import time

env = gym.make("FetchPickAndPlaceEnv-v0")

for e in range(10):
    env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
