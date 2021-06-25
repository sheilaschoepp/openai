import math
import os
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import load_model_from_path, MjSim, functions
from tqdm import tqdm

MODEL_XML = "/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"

model = load_model_from_path(MODEL_XML)
sim = MjSim(model)

# joints

# robot0:slide0, no range

# robot0:slide1, no range

# robot0:slide2, no range

# robot0:torso_lift_joint, range 0.0386 0.3861
# torso_lift_joint = np.concatenate((np.arange(0.0386,  0.38, 0.1), np.full((1,), 0.3861)))
torso_lift_joint = np.linspace(start=0.0386, stop=0.3861, num=10)

# robot0:head_pan_joint, range -1.57 1.57
# head_pan_joint = np.concatenate((np.arange(-1.57, 1.57, 0.1), np.full((1,), 1.57)))
head_pan_joint = np.linspace(start=-1.57, stop=1.57, num=10)

# robot0:head_tilt_joint, range -0.76 1.45
# head_tilt_joint = np.concatenate((np.arange(-0.76, 1.45, 0.1), np.full((1,), 1.45)))
head_tilt_joint = np.linspace(start=-0.76, stop=1.45, num=10)

# robot0:shoulder_pan_joint, range -1.6056 1.6056
# shoulder_pan_joint = np.concatenate((np.arange(-1.6056, 1.61, 0.1), np.full((1,), 1.6057)))
shoulder_pan_joint = np.linspace(start=-1.6056, stop=1.6056, num=10)

# robot0:shoulder_lift_joint, range -1.221 1.518
# shoulder_lift_joint = np.concatenate((np.arange(-1.221, 1.518, 0.1), np.full((1,), 1.518)))
shoulder_lift_joint = np.linspace(start=-1.221, stop=1.518, num=10)

# robot0:upperarm_roll_joint, no range

# robot0:elbow_flex_joint, range -2.251 2.251
# elbow_flex_joint = np.concatenate((np.arange(-2.251, 2.25, 0.1), np.full((1,), 2.251)))
elbow_flex_joint = np.linspace(start=-2.251, stop=2.251, num=10)

# robot0:forearm_roll_joint, no range

# robot0:wrist_flex_joint, range -2.16 2.16
# wrist_flex_joint = np.concatenate((np.arange(-2.16, 2.17, 0.1), np.full((1,), 2.16)))
wrist_flex_joint = np.linspace(start=-2.16, stop=2.16, num=10)

# robot0:wrist_roll_joint, no range

# robot0:r_gripper_finger_joint, range 0 0.05
# r_gripper_finger_joint = np.arange(0, 0.06, 0.025)

# robot0:l_gripper_finger_joint, range 0 0.05
# l_gripper_finger_joint = np.arange(0, 0.06, 0.025)

num_points = torso_lift_joint.shape[0] * \
             head_pan_joint.shape[0] * \
             head_tilt_joint.shape[0] * \
             shoulder_pan_joint.shape[0] * \
             shoulder_lift_joint.shape[0] * \
             elbow_flex_joint.shape[0] * \
             wrist_flex_joint.shape[0]  #* \
             # r_gripper_finger_joint.shape[0] * \
             # l_gripper_finger_joint.shape[0]

print(num_points)
# 8258144400
# 10000000

points = []

functions.mj_kinematics(model, sim.data)  # run forward kinematics, returns None
functions.mj_forward(model, sim.data)  # same as mj_step but does not integrate in time, returns None

with tqdm(total=num_points) as pbar:
    for i in torso_lift_joint:
        sim.data.set_joint_qpos("robot0:torso_lift_joint", i)

        for j in head_pan_joint:
            sim.data.set_joint_qpos("robot0:head_pan_joint", j)

            for k in head_tilt_joint:
                sim.data.set_joint_qpos("robot0:head_tilt_joint", k)

                for l in shoulder_pan_joint:
                    sim.data.set_joint_qpos("robot0:shoulder_pan_joint", l)

                    for m in shoulder_lift_joint:
                        sim.data.set_joint_qpos("robot0:shoulder_lift_joint", m)

                        for n in elbow_flex_joint:
                            sim.data.set_joint_qpos("robot0:elbow_flex_joint", n)

                            for o in wrist_flex_joint:
                                sim.data.set_joint_qpos("robot0:wrist_flex_joint", o)

                                # important: removed the next two joints from the computation since they have no effect on grip position

                                # for p in r_gripper_finger_joint:
                                #     sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", p)
                                #
                                #     for q in l_gripper_finger_joint:
                                #         sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", q)

                                functions.mj_kinematics(model, sim.data)
                                functions.mj_forward(model, sim.data)

                                point = sim.data.get_site_xpos("robot0:grip").copy()  # must use copy here; otherwise all points in list are same
                                points.append(point)

                                # print(point)

                                pbar.update(1)

# data

print("points", len(points))
unique_points = np.unique(points, axis=0)
print("unique", len(unique_points))

data_directory = os.getcwd() + "/data"
os.makedirs(data_directory, exist_ok=True)

# np.save(data_directory + "/points.npy", points)
np.savetxt(data_directory + "/points.csv", points, delimiter=",")
# np.save(data_directory + "/unique_points.npy", unique_points)
np.savetxt(data_directory + "/unique_points.csv", unique_points, delimiter=",")

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

plt.savefig(plot_directory + "/robot_workspace.jpg")
# plt.show()
