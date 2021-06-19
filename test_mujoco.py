from mujoco_py import load_model_from_path, MjSim, functions
import numpy as np
MODEL_XML = "/home/mehran/Documents/openai/custom_gym_envs/envs/FetchReach/Normal/assets/fetch/reach.xml"
model = load_model_from_path(MODEL_XML)
sim = MjSim(model)
data = sim.data # not sure if this is right
print(functions.mj_kinematics(model, data)) # this doesn't work - shows None