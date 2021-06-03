#Importing OpenAI gym package and MuJoCo engine
import gym
from environment.environment import Environment
import mujoco_py
import custom_gym_envs
#Setting MountainCar-v0 as the environment
# env = Environment('FetchPickAndPlace-v0', 0)
env = gym.make('FetchPickAndPlaceDense-v0')
# env = gym.make('AntEnv-v0')

#Sets an initial state
state = env.reset()
# env.render()
# Rendering our instance 300 times
# print(state.shape)
# print(env.observation_space)
# print(env.observation_space['achieved_goal'].shape)
# print(env.observation_space['desired_goal'].shape)
# print(env.observation_space['observation'].shape)

# print(env.env_state_dim())

counter = 0
while True:
    #renders the environment
    env.render()
    #Takes a random action from its action space
    # aka the number of unique actions an agent can perform
    state, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    # print('#' * 50)
    # print(state[0]['observation'])
    # print(state[0]['achieved_goal'])
    # print(state[0]['desired_goal'])
    #
    # print('#' * 50)

    # counter += 1
    # if counter == 100:
    #     counter = 0
    #     env.reset()

env.close()