import numpy as np
import torch
import random

SEED = 0

# -----------------------------------------------------------
# Set and Save Initial State.
# -----------------------------------------------------------

# Set seeds for reproducibility.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Save initial Python, NumPy, and Torch RNG states.
python_random_state = random.getstate()
numpy_random_state = np.random.get_state()
torch_random_state = torch.get_rng_state()

# Sample random values.
saved_random_value = random.random()
saved_numpy_value = np.random.rand()
saved_torch_value = torch.rand(1).item()

# -----------------------------------------------------------
# Restore State and Test Consistency.
# -----------------------------------------------------------

# Restore Python, NumPy, and Torch RNG states.
random.setstate(python_random_state)
np.random.set_state(numpy_random_state)
torch.set_rng_state(torch_random_state)

# Sample random values after restoring the RNG states.
restored_random_value = random.random()
restored_numpy_value = np.random.rand()
restored_torch_value = torch.rand(1).item()

# -----------------------------------------------------------
# Tests for Random, NumPy, and Torch Consistency.
# -----------------------------------------------------------

print("Are Python random values equal?", saved_random_value == restored_random_value)

print("Are NumPy random values equal?", np.isclose(saved_numpy_value, restored_numpy_value))

print("Are Torch random values equal?", np.isclose(saved_torch_value, restored_torch_value))