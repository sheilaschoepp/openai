import gymnasium as gym
import numpy as np
import torch
import random

SEED = 0

# Initialize environment.
env = gym.make("Ant-v5")

# -----------------------------------------------------------
# Set and Save Initial State.
# -----------------------------------------------------------

# Seed Gym environment.
observation, info = env.reset(seed=SEED)
env.action_space.seed(SEED)

# Save the initial observation.
saved_observation = env.unwrapped._get_obs()

# Save MuJoCo simulation state: qpos and qvel.
mujoco_qpos = env.unwrapped.data.qpos.copy()
mujoco_qvel = env.unwrapped.data.qvel.copy()

# Save Gymnasium environment internal RNG state.
gym_random_state = env.np_random.bit_generator.state

# Save action space RNG state.
action_space_random_state = env.action_space.np_random.bit_generator.state

# Sample an action.
saved_action = env.action_space.sample()

# Take a step in the environment to change the state.
env.step(saved_action)

# -----------------------------------------------------------
# Restore State and Test Consistency
# -----------------------------------------------------------

# Reset the Gym environment with a different seed.
# Note: The seed here doesn't matter since we'll restore the state.
env.reset(seed=2)

# Restore MuJoCo simulation state.
env.unwrapped.set_state(mujoco_qpos, mujoco_qvel)

# Restore Gymnasium RNG states.
env.np_random.bit_generator.state = gym_random_state
env.action_space.np_random.bit_generator.state = action_space_random_state

# Retrieve the restored observation.
restored_observation = env.unwrapped._get_obs()

# Sample an action after restoring the RNG state.
restored_action = env.action_space.sample()

# -----------------------------------------------------------
# Verify Restored Environment State
# -----------------------------------------------------------

# Check if the observations are identical.
observations_equal = np.array_equal(saved_observation, restored_observation)
print("Are observations equal?", observations_equal)

# Check if the actions are identical.
actions_equal = np.array_equal(saved_action, restored_action)
print("Are actions equal?", actions_equal)
