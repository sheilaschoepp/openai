import numpy as np


def rot_xyz_matrix(x_angle, y_angle, z_angle):
    rx = [[ 1.,                   0.,                   0.],
          [ 0.,                   np.cos(x_angle),     -np.sin(x_angle)],
          [ 0.,                   np.sin(x_angle),      np.cos(x_angle)]]
    ry = [[ np.cos(y_angle),      0.,                   np.sin(y_angle)],
          [ 0.,                   1.,                   0.],
          [-np.sin(y_angle),      0.,                   np.cos(y_angle)]]
    rz = [[ np.cos(z_angle),     -np.sin(z_angle),      0.],
          [ np.sin(z_angle),      np.cos(z_angle),      0.],
          [ 0.,                   0.,                   1.]]

    r = np.matmul(np.matmul(rx, ry), rz)

    r = np.append(r, np.zeros((3, 1)), axis=1)
    r = np.append(r, np.zeros((1, 4)), axis=0)
    r[3][3] = 1.

    ro = r

    return ro