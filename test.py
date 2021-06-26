# Importing OpenAI gym package and MuJoCo engine
import gym
from tqdm import tqdm
import numpy as np
from mujoco_py import functions, load_model_from_path
from environment.environment import Environment
import custom_gym_envs


# Setting MountainCar-v0 as the environment
# env = Environment('FetchPickAndPlace-v0', 0)
# env = gym.make('FetchPickAndPlaceDense-v0')
env = gym.make('FetchReach-v0')
# env = gym.make('FetchReachFaultyJoint-v1')
# env = gym.make('FetchReachFaultyJoint-v2')
# env = gym.make('FetchReachFaultyJoint-v3')
# env = gym.make('FetchReachFaultyBrokenGrip-v0')
# env = gym.make('FetchReach-v1')
# env = gym.make('AntEnv-v0')

print(env.observation_space['achieved_goal'])
print(env.observation_space['desired_goal'])
print(env.observation_space['observation'])

env._max_episode_steps = 300
offset = 0.0167
# offset = 0.02
velocity = 0.1
num_points_per_axis = 20
counter = 0
reward = -1
render = True

points = np.zeros([num_points_per_axis ** 3, 4])


def reduce_distance():
    global current_pos, state, reward, done
    action = np.zeros(4)
    while abs(goal[0] - current_pos[0]) > offset and reward != 0 and not done:
        if goal[0] > current_pos[0]:
            action[0] = velocity
        else:
            action[0] = -velocity
        prev_pos = current_pos
        state, reward, done, info = env.step(action)
        current_pos = state['achieved_goal']
        print(abs(prev_pos - current_pos))
        if render:
            env.render()
    action = np.zeros(4)
    while abs(goal[1] - current_pos[1]) > offset and reward != 0 and not done:
        current_pos = state['achieved_goal']
        if goal[1] > current_pos[1]:
            action[1] = velocity
        else:
            action[1] = -velocity
        prev_pos = current_pos
        state, reward, done, info = env.step(action)
        current_pos = state['achieved_goal']
        print(abs(prev_pos - current_pos))
        if render:
            env.render()
    action = np.zeros(4)
    while abs(goal[2] - current_pos[2]) > offset and reward != 0 and not done:
        current_pos = state['achieved_goal']
        prev_pos = current_pos
        if goal[2] > current_pos[2]:
            action[2] = velocity
        else:
            action[2] = -velocity
        prev_pos = current_pos
        state, reward, done, info = env.step(action)
        current_pos = state['achieved_goal']
        print(abs(prev_pos - current_pos))
        if render:
            env.render()


with tqdm(total=num_points_per_axis ** 3) as pbar:
    for x in np.linspace(-0.15, 0.15, num_points_per_axis):
        for y in np.linspace(-0.15, 0.15, num_points_per_axis):
            for z in np.linspace(-0.15, 0.15, num_points_per_axis):
                done = False
                reward = -1
                goal = env.initial_gripper_xpos[:3] + np.array([x, y, z])
                env.set_goal(goal)
                state = env.reset()
                if render:
                    env.render()
                # action = env.action_space.sample()
                goal = state['desired_goal']
                current_pos = state['achieved_goal']

                while (abs(current_pos - goal) > offset).any() and reward != 0 and not done:
                    reduce_distance()

                points[counter, :3] = goal
                points[counter, 3] = (True if (reward == 0) else False)
                counter += 1
                pbar.update(1)

env.close()
