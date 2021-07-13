# Importing OpenAI gym package and MuJoCo engine
import time

import gym
from tqdm import tqdm
import numpy as np
import custom_gym_envs

env = gym.make('FetchReachEnv-v0')

# print(env.observation_space['achieved_goal'])
# print(env.observation_space['desired_goal'])
# print(env.observation_space['observation'])

env._max_episode_steps = 300
offset = 0.001
# offset_original = env.distance_threshold
offset_original = 0.005
# offset = 0.02
num_points_per_axis = 10
counter = 0
render = True

_INITIAL_GRIPPER_POS = np.array([1.34183226, 0.74910038, 0.53472284])
# _INITIAL_GRIPPER_POS = env.initial_gripper_xpos[:3]
points = np.zeros([num_points_per_axis ** 3, 4])
points_false_negative = np.zeros([num_points_per_axis ** 3, 4])


def reduce_distance():
    global current_pos, state, done
    goal = env.goal.copy()
    action = np.zeros(4)
    while np.linalg.norm(goal - current_pos, axis=-1) > offset and not done:
        action[:3] = goal - current_pos
        state, _, done, _ = env.step(action)
        current_pos = state['achieved_goal']
        if render:
            env.render()

    return np.linalg.norm(goal - current_pos, axis=-1) <= offset, \
           np.linalg.norm(goal - current_pos, axis=-1) <= offset_original


with tqdm(total=num_points_per_axis ** 3) as pbar:
    for x in np.linspace(-0.15, 0.15, num_points_per_axis):
        for y in np.linspace(-0.15, 0.15, num_points_per_axis):
            for z in np.linspace(-0.15, 0.15, num_points_per_axis):
                done = False
                goal = _INITIAL_GRIPPER_POS + np.array([x, y, z])

                # state = env.reset()
                # env.set_goal(goal)
                # if render:
                #     env.render()
                # goal = state['desired_goal']
                # current_pos = state['achieved_goal']

                env.reset()
                # goal = [1.19, 0.76, 0.68]
                env.set_goal(goal)
                gripper_target = goal.copy()
                gripper_rotation = np.array([1., 0., 1., 0.])
                env.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
                env.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
                for _ in range(20):
                    env.sim.step()

                # if render:
                env.render()
                time.sleep(2)

                points[counter, :3] = goal
                points_false_negative[counter, :3] = goal

                # points[counter, 3], points_false_negative[counter, 3] = reduce_distance()

                points[counter, 3] = \
                    (np.linalg.norm(goal - env.sim.data.get_site_xpos('robot0:grip'), axis=-1) <= offset)
                points_false_negative[counter, 3] = \
                    (np.linalg.norm(goal - env.sim.data.get_site_xpos('robot0:grip'), axis=-1) <= offset_original)

                counter += 1
                pbar.update(1)

reachable = points[np.where(points[:, -1] == 1)[0], :]
unreachable = points[np.where(points[:, -1] == 0)[0], :]
reachable_false_negative = points_false_negative[np.where(points_false_negative[:, -1] == 1)[0], :]
unreachable_false_negative = points_false_negative[np.where(points_false_negative[:, -1] == 0)[0], :]

print(reachable.shape)
print(unreachable.shape)

print('False Negative')
print(reachable_false_negative.shape)
print(unreachable_false_negative.shape)

env.close()
