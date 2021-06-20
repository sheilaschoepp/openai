# from mujoco_py import load_model_from_path, MjSim, functions
# import numpy as np
# MODEL_XML = "/home/mehran/Documents/openai/custom_gym_envs/envs/FetchReach/Normal/assets/fetch/reach.xml"
# model = load_model_from_path(MODEL_XML)
# sim = MjSim(model)
# data = sim.data # not sure if this is right
#
# while True:
#     # functions.mj_kinematics(model, data)
#     functions.mj_step(model, data)
#     functions.mj_inverse(model, data)
#     print(sim.data.qfrc_inverse)


import gym
import numpy as np
from mujoco_py import functions, load_model_from_path
import custom_gym_envs
env = gym.make('FetchReach-v0')
env.reset()
sim = env.sim
# model = load_model_from_path("/home/mehran/Documents/openai/custom_gym_envs/envs/FetchReach/Normal/assets/fetch/reach.xml")
prev = sim.data.qfrc_inverse


for wrist_flex_joint_angle in np.linspace(-2.16, 2.16, 100):
    for shoulder_lift_joint_angle in np.linspace(-1.221, 1.518, 100):
        for elbow_flex_joint_angle in np.linspace(-2.251, 2.251, 100):
            env.sim.data.set_joint_qpos("robot0:wrist_flex_joint", wrist_flex_joint_angle)
            env.sim.data.set_joint_qpos("robot0:shoulder_lift_joint", shoulder_lift_joint_angle)
            env.sim.data.set_joint_qpos("robot0:elbow_flex_joint", elbow_flex_joint_angle)
            functions.mj_kinematics(env.sim.model, env.sim.data)
            functions.mj_step(sim.model, sim.data)
            # action = env.action_space.sample()
            # env.step(action)
            # functions.mj_inverse(env.sim.model, env.sim.data)
            # print(env.sim.data.get_joint_qpos("robot0:wrist_flex_joint"))
            # print(env.sim.data.get_joint_qpos("robot0:shoulder_lift_joint"))
            # print(env.sim.data.get_joint_qpos("robot0:elbow_flex_joint"))

            env.render()

    # print(sim.data.qfrc_inverse == prev)
    # prev = sim.data.qfrc_inverse.copy()

