import gymnasium as gym
import custom_gym_envs

env = gym.make("AntEnv-F3", render_mode="human")

env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())
