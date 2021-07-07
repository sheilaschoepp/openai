import argparse
import math
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import load_model_from_path, MjSim, functions
from tqdm import tqdm
from pathlib import Path

import custom_gym_envs  # DO NOT DELETE

parser = argparse.ArgumentParser(description="FetchReach Kinematics Arguments")

parser.add_argument("-e", "--env_name", default="FetchReach-v1",
                    help="name of normal (non-malfunctioning) MuJoCo Gym environment (default: FetchReach-v1)")

args = parser.parse_args()

# accuracy level
# pan motion is given a higher accuracy than lift/flex motions

LVL_1 = 5
LVL_2 = 10

ACCURACY_LVL_1 = np.radians(LVL_1)  # radians
ACCURACY_LVL_2 = np.radians(LVL_2)  # radians

# xml

if "melco2" in os.uname()[1]:
    anaconda_path = "/opt/anaconda3"
elif "melco" in os.uname()[1]:
    anaconda_path = "/local/melco2/sschoepp/anaconda3"
else:
    anaconda_path = os.getenv("HOME") + "/anaconda3"

home_dir = str(Path.home())

model_xml = None
if args.env_name == "FetchReach-v1":
    model_xml = anaconda_path + "/envs/openai/lib/python3.9/site-packages/gym/envs/robotics/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v0":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v0_Normal/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v1":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v1_BrokenShoulderLiftJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v2":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v2_BrokenElbowFlexJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v3":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v3_BrokenWristFlexJoint/assets/fetch/reach.xml"
elif args.env_name == "FetchReachEnv-v4":
    model_xml = home_dir + "/Documents/openai/custom_gym_envs/envs/fetchreach/FetchReachEnv_v4_BrokenGrip/assets/fetch/reach.xml"

robot_xml = model_xml[:-9] + "robot.xml"
tree = ET.parse(robot_xml)
root = tree.getroot()

torso_lift_joint_range = None
head_pan_joint_range = None
head_tilt_joint_range = None
shoulder_pan_joint_range = None
shoulder_lift_joint_range = None
elbow_flex_joint_range = None
wrist_flex_joint_range = None
r_gripper_finger_joint_range = None
l_gripper_finger_joint_range = None

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
    elif name == "robot0:r_gripper_finger_joint":
        r_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)
    elif name == "robot0:l_gripper_finger_joint":
        l_gripper_finger_joint_range = np.array(attrib.get("range").split(" "), dtype=float)

upperarm_roll_joint_range = np.array([-np.pi, np.pi])
forearm_roll_joint_range = np.array([-np.pi, np.pi])
wrist_roll_joint_range = np.array([-np.pi, np.pi])

# joint angles

# robot0:slide0, no range

# robot0:slide1, no range

# robot0:slide2, no range

# robot0:torso_lift_joint, range 0.0386 0.3861
num = max(math.ceil((torso_lift_joint_range[1] - torso_lift_joint_range[0]) / ACCURACY_LVL_2), 2)
torso_lift_joint_angles = np.linspace(start=torso_lift_joint_range[0], stop=torso_lift_joint_range[1], num=num)

# robot0:head_pan_joint, range -1.57 1.57
num = max(math.ceil((head_pan_joint_range[1] - head_pan_joint_range[0]) / ACCURACY_LVL_1), 2)
head_pan_joint_angles = np.linspace(start=head_pan_joint_range[0], stop=head_pan_joint_range[1], num=num)

# robot0:head_tilt_joint, range -0.76 1.45
num = max(math.ceil((head_tilt_joint_range[1] - head_tilt_joint_range[0]) / ACCURACY_LVL_2), 2)
head_tilt_joint_angles = np.linspace(start=head_tilt_joint_range[0], stop=head_tilt_joint_range[1], num=num)

# robot0:shoulder_pan_joint, range -1.6056 1.6056
num = max(math.ceil((shoulder_pan_joint_range[1] - shoulder_pan_joint_range[0]) / ACCURACY_LVL_1), 2)
shoulder_pan_joint_angles = np.linspace(start=shoulder_pan_joint_range[0], stop=shoulder_pan_joint_range[1], num=num)

# robot0:shoulder_lift_joint, range -1.221 1.518
num = max(math.ceil((shoulder_lift_joint_range[1] - shoulder_lift_joint_range[0]) / ACCURACY_LVL_2), 2)
shoulder_lift_joint_angles = np.linspace(start=shoulder_lift_joint_range[0], stop=shoulder_lift_joint_range[1], num=num)

# robot0:upperarm_roll_joint, limited=false, range -pi, pi
num = max(math.ceil((upperarm_roll_joint_range[1] - upperarm_roll_joint_range[0]) / ACCURACY_LVL_2), 2)
upperarm_roll_joint_angles = np.linspace(start=upperarm_roll_joint_range[0], stop=upperarm_roll_joint_range[1], num=num)

# robot0:elbow_flex_joint, range -2.251 2.251
num = max(math.ceil((elbow_flex_joint_range[1] - elbow_flex_joint_range[0]) / ACCURACY_LVL_2), 2)
elbow_flex_joint_angles = np.linspace(start=elbow_flex_joint_range[0], stop=elbow_flex_joint_range[1], num=num)

# robot0:forearm_roll_joint, limited=false, range -pi, pi
num = max(math.ceil((forearm_roll_joint_range[1] - forearm_roll_joint_range[0]) / ACCURACY_LVL_2), 2)
forearm_roll_joint_angles = np.linspace(start=forearm_roll_joint_range[0], stop=forearm_roll_joint_range[1], num=num)

# robot0:wrist_flex_joint, range -2.16 2.16
num = max(math.ceil((wrist_flex_joint_range[1] - wrist_flex_joint_range[0]) / ACCURACY_LVL_2), 2)
wrist_flex_joint_angles = np.linspace(start=wrist_flex_joint_range[0], stop=wrist_flex_joint_range[1], num=num)

# robot0:wrist_roll_joint, limited=false, range -pi, pi
num = max(math.ceil((wrist_roll_joint_range[1] - wrist_roll_joint_range[0]) / ACCURACY_LVL_2), 2)
wrist_roll_joint_angles = np.linspace(start=wrist_roll_joint_range[0], stop=wrist_roll_joint_range[1], num=num)

# robot0:r_gripper_finger_joint, range 0 0.05  # not included, no effect on end effector position
num = max(math.ceil((r_gripper_finger_joint_range[1] - r_gripper_finger_joint_range[0]) / ACCURACY_LVL_2), 2)
r_gripper_finger_joint_angles = np.linspace(start=r_gripper_finger_joint_range[0], stop=r_gripper_finger_joint_range[1], num=num)

# robot0:l_gripper_finger_joint, range 0 0.05  # not included, no effect on end effector position
num = max(math.ceil((l_gripper_finger_joint_range[1] - l_gripper_finger_joint_range[0]) / ACCURACY_LVL_2), 2)
l_gripper_finger_joint_angles = np.linspace(start=l_gripper_finger_joint_range[0], stop=l_gripper_finger_joint_range[1], num=num)

# scan through joint angles

model = load_model_from_path(model_xml)
sim = MjSim(model)

num_points = torso_lift_joint_angles.shape[0] * \
             head_pan_joint_angles.shape[0] * \
             head_tilt_joint_angles.shape[0] * \
             shoulder_pan_joint_angles.shape[0] * \
             shoulder_lift_joint_angles.shape[0] * \
             elbow_flex_joint_angles.shape[0] * \
             wrist_flex_joint_angles.shape[0]


def test():
    """
    test how each joint affects the robot0:grip position (use PyCharm debugger)
    """

    functions.mj_kinematics(model, sim.data)  # run forward kinematics, returns None
    functions.mj_forward(model, sim.data)  # same as mj_step but does not integrate in time, returns None

    test_points = []

    for i in torso_lift_joint_angles:  # result: affects the z coordinate robot0:grip
        sim.data.set_joint_qpos("robot0:torso_lift_joint", i)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for j in head_pan_joint_angles:  # result: does not affect robot0:grip position
        sim.data.set_joint_qpos("robot0:head_pan_joint", j)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for k in head_tilt_joint_angles:  # result: does not affect robot0:grip position
        sim.data.set_joint_qpos("robot0:head_tilt_joint", k)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for l in shoulder_pan_joint_angles:  # result: does affect robot0:grip position
        sim.data.set_joint_qpos("robot0:shoulder_pan_joint", l)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for m in shoulder_lift_joint_angles:  # result: does affect robot0:grip position
        sim.data.set_joint_qpos("robot0:shoulder_lift_joint", m)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for n in upperarm_roll_joint_angles:  # result: does affect robot0:grip position (although it is not apparent in the points list)
        sim.data.set_joint_qpos("robot0:upperarm_roll_joint", n)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for o in elbow_flex_joint_angles:  # result: does affect robot0:grip position
        sim.data.set_joint_qpos("robot0:elbow_flex_joint", o)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for p in forearm_roll_joint_angles:  # result: does affect robot0:grip position (although it is not apparent in the points list)
        sim.data.set_joint_qpos("robot0:forearm_roll_joint", p)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for q in wrist_flex_joint_angles:  # result: does affect robot0:grip position
        sim.data.set_joint_qpos("robot0:wrist_flex_joint", q)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []
    
    for r in wrist_roll_joint_angles:  # result: does affect robot0:grip position (although it is not apparent in the points list)
        sim.data.set_joint_qpos("robot0:wrist_roll_joint", r)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for s in r_gripper_finger_joint_angles:  # result: does affect robot0:grip position
        sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", s)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []

    for t in l_gripper_finger_joint_angles:  # result: does affect robot0:grip position
        sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", t)

        functions.mj_kinematics(model, sim.data)
        functions.mj_forward(model, sim.data)

        point = sim.data.get_site_xpos("robot0:grip").copy()
        test_points.append(point)

    test_points = []


# test()


def workspace_points():
    """
    scan through each possible joint angle combination to plot the workspace

    torso_lift_joint_angles i
    head_pan_joint_angles j
    head_tilt_joint_angles k
    shoulder_pan_joint_angles l
    shoulder_lift_joint_angles m
    upperarm_roll_joint_angles n
    elbow_flex_joint_angles o
    forearm_roll_joint_angles p
    wrist_flex_joint_angles q
    wrist_roll_joint_angles r
    r_gripper_finger_joint_angles s
    l_gripper_finger_joint_angles t
    """

    points = []

    functions.mj_kinematics(model, sim.data)  # run forward kinematics, returns None
    functions.mj_forward(model, sim.data)  # same as mj_step but does not integrate in time, returns None

    with tqdm(total=num_points) as pbar:
        for i in torso_lift_joint_angles:
            sim.data.set_joint_qpos("robot0:torso_lift_joint", i)

            for l in shoulder_pan_joint_angles:
                sim.data.set_joint_qpos("robot0:shoulder_pan_joint", l)

                for m in shoulder_lift_joint_angles:
                    sim.data.set_joint_qpos("robot0:shoulder_lift_joint", m)

                    for n in upperarm_roll_joint_angles:
                        sim.data.set_joint_qpos("robot0:upperarm_roll_joint", n)

                        for o in elbow_flex_joint_angles:
                            sim.data.set_joint_qpos("robot0:elbow_flex_joint", o)
                            
                            for p in forearm_roll_joint_angles:
                                sim.data.set_joint_qpos("robot0:forearm_roll_joint", p)
                                
                                for q in wrist_flex_joint_angles:
                                    sim.data.set_joint_qpos("robot0:wrist_flex_joint", q)
                                    
                                    for r in wrist_roll_joint_angles:
                                        sim.data.set_joint_qpos("robot0:wrist_roll_joint", r)

                                        functions.mj_kinematics(model, sim.data)
                                        functions.mj_forward(model, sim.data)
            
                                        point = sim.data.get_site_xpos("robot0:grip").copy()  # must use copy here; otherwise all points in list are same
                                        points.append(point)
            
                                        pbar.update(1)

            points = list(np.unique(points, axis=0))

    print("points", len(points))

    data_directory = os.getcwd() + "/data/{}".format(args.env_name)
    os.makedirs(data_directory, exist_ok=True)

    np.save(data_directory + "/{}_points_{}_{}.npy".format(args.env_name, LVL_1, LVL_2), points)

    return points


def plot(points):
    """
    plot the points generated in the call to the workspace_points method (or the robot's approximate 3D workspace)
    """

    plot_directory = os.getcwd() + "/plots/{}".format(args.env_name)
    os.makedirs(plot_directory, exist_ok=True)

    # set min and max x,y,z to the x,y,z of the first point in list
    min_x = points[0][0]
    min_y = points[0][1]
    min_z = points[0][2]

    max_x = points[0][0]
    max_y = points[0][1]
    max_z = points[0][2]

    for p in points:

        if p[0] < min_x:
            min_x = p[0]
        elif p[0] > max_x:
            max_x = p[0]
        else:
            pass

        if p[1] < min_y:
            min_y = p[1]
        elif p[1] > max_y:
            max_y = p[1]
        else:
            pass

        if p[2] < min_z:
            min_z = p[2]
        elif p[2] > max_z:
            max_z = p[2]
        else:
            pass

    print("x: [{}, {}]".format(min_x, max_x))
    print("y: [{}, {}]".format(min_y, max_y))
    print("z: [{}, {}]".format(min_z, max_z))

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    z_line = np.linspace(start=min_z, stop=max_z, num=10)
    x_line = np.linspace(start=min_x, stop=max_x, num=10)
    y_line = np.linspace(start=min_y, stop=max_y, num=10)
    ax.plot3D(x_line, y_line, z_line, "gray")

    x_points = []
    y_points = []
    z_points = []

    for p in points:
        x_points.append(p[0])
        y_points.append(p[1])
        z_points.append(p[2])

    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap="hsv");

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.savefig(plot_directory + "/{}_workspace_{}_{}.jpg".format(args.env_name, LVL_1, LVL_2))


if __name__ == "__main__":

    wp = workspace_points()
    plot(wp)
