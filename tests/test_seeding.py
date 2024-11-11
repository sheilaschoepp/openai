import gymnasium as gym
import numpy as np
import torch
import random

SEED = 0

"""
Gymnasium
"""

# Create a Gymnasium environment.
env = gym.make("Ant-v5")

# Seed the environment.
observation, info = env.reset(seed=SEED)
env.action_space.seed(SEED)

# -----------------------------------------------------------
# Save the environment.
# -----------------------------------------------------------

# Save the initial observation.
saved_observation = observation.copy()

# Save the simulation state: qpos and qvel.
mujoco_qpos = env.unwrapped.data.qpos.copy()
mujoco_qvel = env.unwrapped.data.qvel.copy()

# Save Gymnasium environment internal RNG state.
gym_random_state = env.np_random.bit_generator.state

# Save the action space RNG state.
action_space_random_state = env.action_space.np_random.bit_generator.state

# Save the first sampled action.
saved_action = env.action_space.sample()

# Take a step in the environment, using the saved action.
# Note: This is to change the observation.
env.step(saved_action)

# -----------------------------------------------------------
# Reset and restore the environment.
# -----------------------------------------------------------

# Reset the Gymnasium environment with a different seed.
# Note: The seed here doesn't matter since we'll restore the state.
env.reset(seed=2)

# Restore the simulation state.
env.unwrapped.set_state(mujoco_qpos, mujoco_qvel)

# Restore Gymnasium RNG state.
env.np_random.bit_generator.state = gym_random_state

# Restore the action space RNG state.
env.action_space.np_random.bit_generator.state = action_space_random_state

# Retrieve the observation after restoring the state.
restored_observation = env.unwrapped._get_obs()

# Sample action after restoring the seed state.
restored_action = env.action_space.sample()

# -----------------------------------------------------------
# Test the restored environment.
# -----------------------------------------------------------

# Check if the observations are identical.
observations_equal = np.array_equal(saved_observation, restored_observation)
print("Are observations equal?", observations_equal)

# Check if the actions are identical.
actions_equal = np.array_equal(saved_action, restored_action)
print("Are actions equal?", actions_equal)







#
# # seed numpy RNG
# np.random.seed(seed_value)
#
# # seed Python's built-in `random` RNG
# random.seed(seed_value)
#
# # seed PyTorch RNG
# torch.manual_seed(seed_value)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed_value)
#
# ### Saving ###
#
#
# # save numpy RNG state
# numpy_random_state = np.random.get_state()
#
# # save PyTorch RNG state
# torch_random_state = torch.get_rng_state()
# if torch.cuda.is_available():
#     torch_cuda_random_state = torch.cuda.get_rng_state()
#
# # save Python's built-in `random` RNG state
# random_state = random.getstate()
#
# # save MuJoCo simulation state if needed
# mujoco_state = None
# if hasattr(env.unwrapped, 'sim'):  # check if MuJoCo sim is accessible
#     mujoco_state = env.unwrapped.sim.get_state()  # Save current MuJoCo simulation state
#
# ### Test ###
#
# action = env.action_space.sample()
# print(action)
#
# ### Loading ###
#
#
#
# # restore numpy RNG state
# np.random.set_state(numpy_random_state)
#
# # restore PyTorch RNG state
# torch.set_rng_state(torch_random_state)
# if torch.cuda.is_available():
#     torch.cuda.set_rng_state(torch_cuda_random_state)
#
# # restore Python `random` RNG state
# random.setstate(random_state)
#
# # restore MuJoCo simulation state if needed
# if mujoco_state is not None and hasattr(env.unwrapped, 'sim'):
#     env.unwrapped.sim.set_state(mujoco_state)
#     env.unwrapped.sim.forward()  # Advance the simulation to apply the state
#
# ### Test ###
#
# action = env.action_space.sample()
# print(action)