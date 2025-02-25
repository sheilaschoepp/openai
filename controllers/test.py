import gymnasium as gym
import time

import custom_gym_envs

env = gym.make("AntEnv-F3-BSL", render_mode="human", camera_name="free")

env.reset(seed=0)

unwrapped_env = env.unwrapped

# Get the correct body ID for "torso"
torso_id = unwrapped_env.model.body(name="torso").id
print(torso_id)

for _ in range(1000):
    env.step(env.action_space.sample())
    time.sleep(0.01)

env.close()
