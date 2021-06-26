import argparse
import math
import multiprocessing as mp
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import load_model_from_path, MjSim, functions
from tqdm import tqdm

import custom_gym_envs  # DO NOT DELETE

parser = argparse.ArgumentParser(description='FetchReach Kinematics Arguments')
parser.add_argument("-e", "--env_name", default="FetchReach-v1",
                    help="name of normal (non-malfunctioning) MuJoCo Gym environment (default: FetchReach-v1)")
args = parser.parse_args()

# xml

MODEL_XML = None
if args.env_name == "FetchReach-v1":
    MODEL_XML = "/opt/anaconda3/envs/openai/lib/python3.7/site-packages/gym/envs/robotics/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v0":
    MODEL_XML = "/opt/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v1":
    MODEL_XML = "/opt/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v2":
    MODEL_XML = "/opt/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v3":
    MODEL_XML = "/opt/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v4":
    MODEL_XML = "/opt/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenGrip/assets/fetch/reach.xml"

ROBOT_XML = MODEL_XML[:-9] + "robot.xml"
tree = ET.parse(ROBOT_XML)
root = tree.getroot()

torso_lift_joint_range = None
head_pan_joint_range = None
head_tilt_joint_range = None
shoulder_pan_joint_range = None
shoulder_lift_joint_range = None
elbow_flex_joint_range = None
wrist_flex_joint_range = None

for child in root.iter():
    attrib = child.attrib
    name = attrib.get("name")
    if name == "robot0:torso_lift_joint":
        torso_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:head_pan_joint":
        head_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:head_tilt_joint":
        head_tilt_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:shoulder_pan_joint":
        shoulder_pan_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:shoulder_lift_joint":
        shoulder_lift_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:elbow_flex_joint":
        elbow_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:wrist_flex_joint":
        wrist_flex_joint_range = np.array(attrib.get("range").split(" "), dtype=float)


# joint angles

ACCURACY = np.radians(5.0)  # radians

# robot0:slide0, no range

# robot0:slide1, no range

# robot0:slide2, no range

# robot0:torso_lift_joint, range 0.0386 0.3861
num = max(math.ceil((torso_lift_joint_range[1] - torso_lift_joint_range[0]) / ACCURACY), 2)
torso_lift_joint_angles = np.linspace(start=torso_lift_joint_range[0], stop=torso_lift_joint_range[1], num=num)

# robot0:head_pan_joint, range -1.57 1.57
num = max(math.ceil((head_pan_joint_range[1] - head_pan_joint_range[0]) / ACCURACY), 2)
head_pan_joint_angles = np.linspace(start=head_pan_joint_range[0], stop=head_pan_joint_range[1], num=num)

# robot0:head_tilt_joint, range -0.76 1.45
num = max(math.ceil((head_tilt_joint_range[1] - head_tilt_joint_range[0]) / ACCURACY), 2)
head_tilt_joint_angles = np.linspace(start=head_tilt_joint_range[0], stop=head_tilt_joint_range[1], num=num)

# robot0:shoulder_pan_joint, range -1.6056 1.6056
num = max(math.ceil((shoulder_pan_joint_range[1] - shoulder_pan_joint_range[0]) / ACCURACY), 2)
shoulder_pan_joint_angles = np.linspace(start=shoulder_pan_joint_range[0], stop=shoulder_pan_joint_range[1], num=num)

# robot0:shoulder_lift_joint, range -1.221 1.518
num = max(math.ceil((shoulder_lift_joint_range[1] - shoulder_lift_joint_range[0]) / ACCURACY), 2)
shoulder_lift_joint_angles = np.linspace(start=shoulder_lift_joint_range[0], stop=shoulder_lift_joint_range[1], num=num)

# robot0:upperarm_roll_joint, no range

# robot0:elbow_flex_joint, range -2.251 2.251
num = max(math.ceil((elbow_flex_joint_range[1] - elbow_flex_joint_range[0]) / ACCURACY), 2)
elbow_flex_joint_angles = np.linspace(start=elbow_flex_joint_range[0], stop=elbow_flex_joint_range[1], num=num)

# robot0:forearm_roll_joint, no range

# robot0:wrist_flex_joint, range -2.16 2.16
num = max(math.ceil((wrist_flex_joint_range[1] - wrist_flex_joint_range[0]) / ACCURACY), 2)
wrist_flex_joint_angles = np.linspace(start=wrist_flex_joint_range[0], stop=wrist_flex_joint_range[1], num=num)

# robot0:wrist_roll_joint, no range

# robot0:r_gripper_finger_joint, range 0 0.05  # not included, no effect on end effector position

# robot0:l_gripper_finger_joint, range 0 0.05  # not included, no effect on end effector position


# scan through joint angles

model = load_model_from_path(MODEL_XML)
sim = MjSim(model)

num_points = torso_lift_joint_angles.shape[0] * \
             head_pan_joint_angles.shape[0] * \
             head_tilt_joint_angles.shape[0] * \
             shoulder_pan_joint_angles.shape[0] * \
             shoulder_lift_joint_angles.shape[0] * \
             elbow_flex_joint_angles.shape[0] * \
             wrist_flex_joint_angles.shape[0]

points = []

functions.mj_kinematics(model, sim.data)  # run forward kinematics, returns None
functions.mj_forward(model, sim.data)  # same as mj_step but does not integrate in time, returns None

with tqdm(total=num_points) as pbar:
    for i in torso_lift_joint_angles:
        sim.data.set_joint_qpos("robot0:torso_lift_joint", i)

        for j in head_pan_joint_angles:
            sim.data.set_joint_qpos("robot0:head_pan_joint", j)

            for k in head_tilt_joint_angles:
                sim.data.set_joint_qpos("robot0:head_tilt_joint", k)

                for l in shoulder_pan_joint_angles:
                    sim.data.set_joint_qpos("robot0:shoulder_pan_joint", l)

                    for m in shoulder_lift_joint_angles:
                        sim.data.set_joint_qpos("robot0:shoulder_lift_joint", m)

                        for n in elbow_flex_joint_angles:
                            sim.data.set_joint_qpos("robot0:elbow_flex_joint", n)

                            for o in wrist_flex_joint_angles:
                                sim.data.set_joint_qpos("robot0:wrist_flex_joint", o)

                                functions.mj_kinematics(model, sim.data)
                                functions.mj_forward(model, sim.data)

                                point = sim.data.get_site_xpos("robot0:grip").copy()  # must use copy here; otherwise all points in list are same
                                points.append(point)

                                pbar.update(1)

# data

print("points", len(points))
unique_points = np.unique(points, axis=0)
print("unique", len(unique_points))

data_directory = os.getcwd() + "/data"
os.makedirs(data_directory, exist_ok=True)

# np.save(data_directory + "/points.npy", points)
np.savetxt(data_directory + "/workspace_all_points.csv", points, delimiter=",")
# np.save(data_directory + "/unique_points.npy", unique_points)
np.savetxt(data_directory + "/workspace_unique_points.csv", unique_points, delimiter=",")

# plot

plot_directory = os.getcwd() + "/plot"
os.makedirs(plot_directory, exist_ok=True)

# set min and max x,y,z to the x,y,z of the first point in list
min_x = unique_points[0][0]
min_y = unique_points[0][1]
min_z = unique_points[0][2]

max_x = unique_points[0][0]
max_y = unique_points[0][1]
max_z = unique_points[0][2]

for up in unique_points:

    if up[0] < min_x:
        min_x = up[0]
    elif up[0] > max_x:
        max_x = up[0]
    else:
        pass

    if up[1] < min_y:
        min_y = up[1]
    elif up[1] > max_y:
        max_y = up[1]
    else:
        pass

    if up[2] < min_z:
        min_z = up[2]
    elif up[2] > max_z:
        max_z = up[2]
    else:
        pass

print("x: [{}, {}]".format(min_x, max_x))
print("y: [{}, {}]".format(min_y, max_y))
print("z: [{}, {}]".format(min_z, max_z))

# important: this is for num=2 so it may not be accurate!
# x: [0.30064826525514765, 0.32427078476066545]
# y: [-0.2773406216316236, 0.8055406216316235]
# z: [0.3371634300212859, 1.7125604647595976]

fig = plt.figure()
ax = plt.axes(projection="3d")

z_line = np.linspace(start=min_z, stop=max_z, num=10)
x_line = np.linspace(start=min_x, stop=max_x, num=10)
y_line = np.linspace(start=min_y, stop=max_y, num=10)
ax.plot3D(x_line, y_line, z_line, 'gray')

x_points = []
y_points = []
z_points = []

for up in unique_points:
    x_points.append(up[0])
    y_points.append(up[1])
    z_points.append(up[2])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig(plot_directory + "/robot_workspace.jpg")
# plt.show()
