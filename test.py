import custom_gym_envs
import gym
import time

from environment.fetch_reach_observation_wrapper import FetchReachObservationWrapper

env = gym.make("FetchReach-v1")




# action space: Box(-1.0, 1.0, (4,), float32)
# observation_space["observation"]: Box(-inf, inf, (25,), float32)
# observation_space["achieved_goal"]: Box(-inf, inf, (3,), float32)
# env.observation_space["desired_goal"]: Box(-inf, inf, (3,), float32)

for e in range(10):
    env.reset()
    done = False
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
