import numpy as np

from kinematics.modified_python.dh import dh
from kinematics.modified_python.rot_xyz_matrix import rot_xyz_matrix

# global variables
SHOULDER_OFFSET_Y = 98. / 1000
LOWER_ARM_LENGTH = 55.95 / 1000
ELBOW_OFFSET_Y = 15. / 1000
SHOULDER_OFFSET_Z = 100. / 1000
HAND_OFFSET_X = 57.75 / 1000
UPPER_ARM_LENGTH = 105. / 1000
HAND_OFFSET_Z = 12.31 / 1000

# added by me: measurements from http://doc.aldebaran.com/2-8/family/nao_technical/links_naov6.html
X_R_WRIST_YAW_to_RFinger11 = 69.07
X_RFinger11_to_RFinger12 = 14.36
X_RFinger12_to_RFinger13 = 14.36
X_RFinger13_to_FingerTip = 14.36  # unpublished

Y_R_WRIST_YAW_to_RFinger11 = 11.57
Y_RFinger11_to_RFinger12 = 0.0
Y_RFinger12_to_RFinger13 = 0.0
Y_RFinger13_to_FingerTip = 0.0  # unpublished

Z_R_WRIST_YAW_to_RFinger11 = -3.04
Z_RFinger11_to_RFinger12 = 0.0
Z_RFinger12_to_RFinger13 = 0.0
Z_RFinger13_to_FingerTip = 0.0  # unpublished

# added by me: calculations for finger offset
FINGER_OFFSET_X = (X_R_WRIST_YAW_to_RFinger11 + X_RFinger11_to_RFinger12 + X_RFinger12_to_RFinger13 + X_RFinger13_to_FingerTip) / 1000
FINGER_OFFSET_Y = (Y_R_WRIST_YAW_to_RFinger11 + Y_RFinger11_to_RFinger12 + Y_RFinger12_to_RFinger13 + Y_RFinger13_to_FingerTip) / 1000
FINGER_OFFSET_Z = (Z_R_WRIST_YAW_to_RFinger11 + Z_RFinger11_to_RFinger12 + Z_RFinger12_to_RFinger13 + Z_RFinger13_to_FingerTip) / 1000


def f_right_hand_h25(thetas):

    assert len(thetas == 5), "f_right_hand_h25 takes 5 arguments"

    base = np.identity(4)
    base[1][3] = -SHOULDER_OFFSET_Y
    base[2][3] = SHOULDER_OFFSET_Z

    t1 = dh(0., -np.pi/2., 0., thetas[0])
    t2 = dh(0., np.pi/2., 0., thetas[1] + np.pi/2.)
    t3 = dh(-ELBOW_OFFSET_Y, np.pi/2., UPPER_ARM_LENGTH, thetas[2])
    t4 = dh(0., -np.pi/2., 0., thetas[3])
    t5 = dh(0., np.pi/2., 0., thetas[4])

    tend1 = np.identity(4)
    # tend1[0][3] = LOWER_ARM_LENGTH + HAND_OFFSET_X
    # tend1[2][3] = -HAND_OFFSET_Z
    # added by me: updated calculations to calculate kinematics to fingertip
    tend1[0][3] = LOWER_ARM_LENGTH + FINGER_OFFSET_X
    tend1[1][3] = FINGER_OFFSET_Y
    tend1[2][3] = FINGER_OFFSET_Z

    r = rot_xyz_matrix(-np.pi/2., 0., -np.pi/2.)
    tend = np.matmul(r, tend1)
    tendend = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(base, t1), t2), t3), t4), t5), tend)

    rot_z = np.arctan2(tendend[1][0], tendend[0][0])
    rot_y = np.arctan2(-tendend[2][0], np.sqrt(tendend[2][1]**2 + tendend[2][2]**2))
    rot_x = np.arctan2(tendend[2][1], tendend[2][2])
    right = np.concatenate((tendend[0:3, 3], [rot_x], [rot_y], [rot_z]))

    # [x, y, z, rot_x, rot_y, rot_z]
    return right


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    print(f_right_hand_h25([0., 0., 0., 0., 0.]))