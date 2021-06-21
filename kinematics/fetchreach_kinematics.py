from mujoco_py import load_model_from_path, MjSim, functions
import numpy as np
from tqdm import tqdm

MODEL_XML = "/home/sschoepp/Documents/openai/custom_gym_envs/envs/fetchreach/Normal/assets/fetch/reach.xml"

model = load_model_from_path(MODEL_XML)
sim = MjSim(model)

# joints
# robot0:slide0, no range
# robot0:slide1, no range
# robot0:slide2, no range
# robot0:torso_lift_joint, range 0.0386 0.3861
# torso_lift_joint = np.arange(0.0386,  0.3862, 0.0001)
# torso_lift_joint = np.concatenate((np.arange(0.0386,  0.38, 0.01), np.full((1,), 0.3861)))
torso_lift_joint = np.concatenate((np.arange(0.0386,  0.38, 0.1), np.full((1,), 0.3861)))
# robot0:head_pan_joint, range -1.57 1.57
# head_pan_joint = np.arange(-1.57, 1.58, 0.01)
head_pan_joint = np.concatenate((np.arange(-1.57, 1.57, 0.1), np.full((1,), 1.57)))
# robot0:head_tilt_joint, range -0.76 1.45
# head_tilt_joint = np.arange(-0.76, 1.46, 0.01)
head_tilt_joint = np.concatenate((np.arange(-0.76, 1.45, 0.1), np.full((1,), 1.45)))
# robot0:shoulder_pan_joint, range -1.6056 1.6056
# shoulder_pan_joint = np.arange(-1.6056, 1.6057, 0.0001)
# shoulder_pan_joint = np.concatenate((np.arange(-1.6056, 1.61, 0.01), np.full((1,), 1.6057)))
shoulder_pan_joint = np.concatenate((np.arange(-1.6056, 1.61, 0.1), np.full((1,), 1.6057)))
# robot0:shoulder_lift_joint, range -1.221 1.518
# shoulder_lift_joint = np.arange(-1.221, 1.519, 0.001)
# shoulder_lift_joint = np.concatenate((np.arange(-1.221, 1.518, 0.01), np.full((1,), 1.518)))
shoulder_lift_joint = np.concatenate((np.arange(-1.221, 1.518, 0.1), np.full((1,), 1.518)))
# robot0:upperarm_roll_joint, no range
# robot0:elbow_flex_joint, range -2.251 2.251
# elbow_flex_joint = np.arange(-2.251, 2.252, 0.001)
# elbow_flex_joint = np.concatenate((np.arange(-2.251, 2.25, 0.01), np.full((1,), 2.251)))
elbow_flex_joint = np.concatenate((np.arange(-2.251, 2.25, 0.1), np.full((1,), 2.251)))
# robot0:forearm_roll_joint, no range
# robot0:wrist_flex_joint, range -2.16 2.16
# wrist_flex_joint = np.arange(-2.16, 2.17, 0.01)
wrist_flex_joint = np.concatenate((np.arange(-2.16, 2.17, 0.1), np.full((1,), 2.16)))
# robot0:wrist_roll_joint, no range
# robot0:r_gripper_finger_joint, range 0 0.05
# r_gripper_finger_joint = np.arange(0, 0.06, 0.01)
r_gripper_finger_joint = np.arange(0, 0.06, 0.025)
# robot0:l_gripper_finger_joint, range 0 0.05
# l_gripper_finger_joint = np.arange(0, 0.06, 0.01)
l_gripper_finger_joint = np.arange(0, 0.06, 0.025)

num_points = torso_lift_joint.shape[0] * \
             head_pan_joint.shape[0] * \
             head_tilt_joint.shape[0] * \
             shoulder_pan_joint.shape[0] * \
             shoulder_lift_joint.shape[0] * \
             elbow_flex_joint.shape[0] * \
             wrist_flex_joint.shape[0] * \
             r_gripper_finger_joint.shape[0] * \
             l_gripper_finger_joint.shape[0]

print(num_points)
# 1506064636361423399247360
# 1580544056307110400
# 74323299600

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

                                for p in r_gripper_finger_joint:
                                    sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", p)

                                    for q in l_gripper_finger_joint:
                                        sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", q)

                                        functions.mj_kinematics(model, sim.data)
                                        functions.mj_forward(model, sim.data)

                                        point = sim.data.get_site_xpos("robot0:grip")
                                        points.append(point)

                                        pbar.update(1)

print("points", len(points))
unique = np.unique(points, axis=0)
print("unique", len(unique))

np.save("data.npy", unique)