#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
import custom_gym_envs
#Setting MountainCar-v0 as the environment
env = gym.make('HumanoidEnv-v1')
# env = gym.make('AntEnv-v0')

#Sets an initial state
env.reset()
# env.render()
# Rendering our instance 300 times
while True:
    #renders the environment
    env.render()
    #Takes a random action from its action space
    # aka the number of unique actions an agent can perform
    env.step(env.action_space.sample())

env.close()