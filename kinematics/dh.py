import numpy as np


def dh(a, alpha, d, theta):
    taf = [[np.cos(theta),              -np.sin(theta),                0.,             a],
           [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
           [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
           [0.,                          0.,                           0.,             1.]]
    return taf


# added by me
def dh2(a, alpha, d, theta):
    taf = [[np.cos(theta),      -np.sin(theta)*np.cos(alpha),    np.sin(theta)*np.sin(alpha),       a*np.cos(theta)],
           [np.sin(theta),       np.cos(theta)*np.cos(alpha),   -np.cos(theta)*np.sin(alpha),       a*np.sin(theta)],
           [0.,                  np.sin(alpha),                  np.cos(alpha),                     d],
           [0.,                  0.,                             0.,                                1.]]
    return taf