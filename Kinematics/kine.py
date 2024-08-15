import numpy as np
from Kinematics.curve_V2 import curve
def kine(pose):
    length = 1
    t = np.linspace(0, length, 100)
    end_point = [0, 0, 0]
    matrix = np.eye(3)
    phi = pose[:3]
    theta = pose[3:6]
    s_phi = 0
    for i in range(3):
        curve_data, end_point, matrix, s_phi = curve(t, end_point, matrix, phi[i], theta[i], s_phi)
        if i == 2:
            return curve_data[:, -1]