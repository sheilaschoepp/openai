from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import custom_gym_envs
import gym
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os, argparse
import logging

import concurrent
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

logging.getLogger().setLevel(logging.ERROR)

mjlib = mjbindings.mjlib

parser = argparse.ArgumentParser(description='FetchReach Kinematics Arguments')

parser.add_argument("-e", "--env_name", default="FetchReachEnv-v0",
                    help="name of normal (non-malfunctioning) MuJoCo Gym environment (default: FetchReachEnv-v0)")

args = parser.parse_args()

if "melco2" in os.uname()[1]:
    anaconda_path = "/opt/anaconda3"
elif "melco" in os.uname()[1]:
    anaconda_path = "/local/melco2/sschoepp/anaconda3"
else:
    anaconda_path = os.getenv("HOME") + "/anaconda3"

home_dir = str(Path.home())

model_xml = None
if args.env_name == "FetchReach-v1":
    model_xml = anaconda_path + "/envs/openai/lib/python3.7/site-packages/gym/envs/robotics/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v0":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/Inverse_kinematics_assets/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v1":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/Inverse_kinematics_assets/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v2":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/Inverse_kinematics_assets/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v3":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/Inverse_kinematics_assets/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v4":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/Inverse_kinematics_assets/FetchReachEnv_v4_BrokenGrip/assets/fetch/reach.xml"

FETCHREACH_XML = assets.get_contents(model_xml)

_SITE_NAME = 'robot0:grip'
_JOINTS = ["robot0:torso_lift_joint",
           "robot0:head_pan_joint",
           "robot0:head_tilt_joint",
           "robot0:shoulder_pan_joint",
           "robot0:shoulder_lift_joint",
           "robot0:elbow_flex_joint",
           "robot0:wrist_flex_joint"]

env = gym.make(args.env_name)

# _INITIAL_GRIP_POS = env.initial_gripper_xpos[:3]
# print(_INITIAL_GRIP_POS)
_INITIAL_GRIP_POS = [1.34183226, 0.74910038, 0.53472284]
_TOL = 1e-14
_MAX_STEPS = 50
_MAX_RESETS = 10
NUM_POINTS_PER_AXIS = 30


class _ResetArm:

    def __init__(self, seed=None):
        self._rng = np.random.RandomState(seed)
        self._lower = None
        self._upper = None

    def _cache_bounds(self, physics):
        self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
        limited = physics.named.model.jnt_limited[_JOINTS].astype(np.bool)
        # Positions for hinge joints without limits are sampled between 0 and 2pi
        self._lower[~limited] = 0
        self._upper[~limited] = 2 * np.pi

    def __call__(self, physics):
        if self._lower is None:
            self._cache_bounds(physics)
        # NB: This won't work for joints with > 1 DOF
        new_qpos = self._rng.uniform(self._lower, self._upper)
        physics.named.data.qpos[_JOINTS] = new_qpos


def test_qpos_from_grip_pose(target, inplace):
    physics = mujoco.Physics.from_xml_string(FETCHREACH_XML)
    target_pos, target_quat = target, None
    count = 0
    physics2 = physics.copy(share_model=True)
    resetter = _ResetArm(seed=0)
    while True:
        result = ik.qpos_from_site_pose(
            physics=physics2,
            site_name=_SITE_NAME,
            target_pos=target_pos,
            target_quat=target_quat,
            joint_names=_JOINTS,
            tol=_TOL,
            max_steps=_MAX_STEPS,
            inplace=inplace,
        )
        if result.success:
            break
        elif count < _MAX_RESETS:
            resetter(physics2)
            count += 1
        else:
            return False

    assert result.steps <= _MAX_STEPS
    assert result.err_norm <= _TOL
    physics.data.qpos[:] = result.qpos
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
    pos = physics.named.data.site_xpos[_SITE_NAME]
    # np.testing.assert_array_almost_equal(pos, target_pos)
    return np.allclose(pos, target_pos, rtol=1e-6)


index = 0
# range = 1.0
range = 0.15


def process_method(x):
    points = []
    for y in np.linspace(-range, range, NUM_POINTS_PER_AXIS):
        for z in np.linspace(-range, range, NUM_POINTS_PER_AXIS):
            single_point = np.zeros([4])
            target = np.array([x, y, z]) + _INITIAL_GRIP_POS
            single_point[:3] = target
            single_point[3] = test_qpos_from_grip_pose(target, False)
            points.append(single_point)

    return points


points = []
print(datetime.now().time())
# with tqdm(total=NUM_POINTS_PER_AXIS ** 3) as pbar:
with ProcessPoolExecutor() as executor:
    results = []
    for x in np.linspace(-range, range, NUM_POINTS_PER_AXIS):
        results.append(executor.submit(process_method, x))

    for r in concurrent.futures.as_completed(results):
        points += r.result()

print(datetime.now().time())
points = np.array(points)

reachable_index = np.where(points[:, 3])
reachable_x = points[reachable_index][:, 0]
reachable_y = points[reachable_index][:, 1]
reachable_z = points[reachable_index][:, 2]

unreachable_index = np.where(points[:, 3] == 0)
unreachable_x = points[unreachable_index][:, 0]
unreachable_y = points[unreachable_index][:, 1]
unreachable_z = points[unreachable_index][:, 2]

path = str(os.path.abspath(Path(__file__).parent)) + f'/data/{args.env_name}/'
if not os.path.exists(path):
    os.mkdir(path)

np.savetxt(path + 'goal_points.csv', points)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(reachable_x, reachable_y, reachable_z, c='g')
ax.scatter(unreachable_x, unreachable_y, unreachable_z, c='r')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

path = str(os.path.abspath(Path(__file__).parent)) + f'/plot/{args.env_name}/'
if not os.path.exists(path):
    os.mkdir(path)

plt.savefig(path + 'reachable_goals.jpg', dpi=300)
plt.show()
