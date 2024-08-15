import numpy as np
import math
# 2D运动学
def kine2D(theta):
    length = 1
    t = np.linspace(0, length, 100)
    end_point = np.array([0, 0])
    matrix = np.eye(2)
    curves_data = []

    for i in range(3):
        # theta 是曲率，1/r
        curve_data = np.zeros((2, len(t)))
        for j, ti in enumerate(t):
            if theta[i] != 0:
                curve_data[1, j] = math.sin(ti * theta[i]) / theta[i]
                curve_data[0, j] = (1 - math.cos(ti * theta[i])) / theta[i]
            else:
                curve_data[1, j] = ti
        for j, ti in enumerate(t):
            curve_data[:, j] = np.dot(matrix, curve_data[:, j])
            curve_data[:, j] += end_point

        w = curve_data[:, -1] - curve_data[:, -2]
        w /= np.linalg.norm(w)  # Normalize w
        v = np.array([-w[1], w[0]])
        matrix = np.column_stack((v, w))

        end_point = curve_data[:, -1]
        curves_data.append(curve_data)

    return curves_data

# print(L.shape)
# (3, 2, 100)