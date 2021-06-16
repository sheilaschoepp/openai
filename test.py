#Importing OpenAI gym package and MuJoCo engine
import gym
import time
import numpy as np
from environment.environment import Environment
import mujoco_py
import custom_gym_envs
#Setting MountainCar-v0 as the environment
# env = Environment('FetchPickAndPlace-v0', 0)
env = gym.make('FetchPickAndPlaceDense-v0')
# env = gym.make('FetchReach-v0')
# env = gym.make('FetchReachFaultyJoint-v1')
# env = gym.make('FetchReachFaultyJoint-v2')
# env = gym.make('FetchReachFaultyJoint-v3')
# env = gym.make('FetchReachFaultyBrokenGrip-v0')

# env = gym.make('FetchReach-v1')

# env = gym.make('AntEnv-v0')

#Sets an initial state
prev_state = env.reset()
# env.render()
# Rendering our instance 300 times
# print(state.shape)
# print(env.observation_space)
print(env.observation_space['achieved_goal'])
print(env.observation_space['desired_goal'])
print(env.observation_space['observation'])

# print(env.env_state_dim())
# print(env.action_space)

reward = 0
# counter = 0
while True:
    #renders the environment
    env.render()

    #Takes a random action from its action space
    # aka the number of unique actions an agent can perform
    # action = env.action_space.sample()
    if reward == -100:
        action = np.array([0.01, 0.01, 0.1, 0.001])
    else:
        action = np.array([0.01, 0.01, -0.1, 0.001])

    # print(action)
    state, reward, done, info = env.step(action)
    # print('reward: ', reward)
    # print(state[np.where(prev_state == state)])
    # print(state['observation'])
    # print(state['achieved_goal'])
    # print(state['desired_goal'])

    # print('#' * 50)
    # print(state[0]['observation'])
    # print(state[0]['achieved_goal'])
    # print(state[0]['desired_goal'])
    # print('#' * 50)
    # time.sleep(3)
    # counter += 1
    # if counter == 50:
    #     counter = 0
    #     env.reset()

env.close()