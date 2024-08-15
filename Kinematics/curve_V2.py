import numpy as np
import math

def curve(t, end_point, matrix, phi=0, theta=0, s_phi = 0):
    #theta是曲率，1/r
    curve_data = np.zeros((3, len(t)))

    for i, ti in enumerate(t):
        if (theta != 0):
            curve_data[2,i] = math.sin(ti * theta) / theta
            c = (1 - math.cos(ti * theta)) / theta
            curve_data[0,i] = c * math.cos(phi)
            curve_data[1,i] = c * math.sin(phi)
        else:
            curve_data[2,i] = ti
    # print("end point = ", end_point)
    for i, ti in enumerate(t):
        curve_data[:,i] = np.dot(matrix,curve_data[:,i])
        curve_data[:,i] += end_point

    # Compute the tangent vector at the curve's end
    w = curve_data[:, -1] - curve_data[:, -2]
    w /= np.linalg.norm(w)  # Normalize w
    # v = np.array([w[2], 0, -w[0]]) 
    #以前用了奇怪的正交，物理上不好解释，现在直接让上级phi指向下级x
    #考虑传递性，phi为0产生麻烦
    s_phi += phi
    z = -w[0]*math.cos(s_phi)-w[1]*math.sin(s_phi)
    v = np.array([math.cos(s_phi),math.sin(s_phi),z])
    v /= np.linalg.norm(v)
    u = np.cross(w, v)
    u /= np.linalg.norm(u)
    matrix = np.column_stack((v, u, w))
    
    end_point = curve_data[:,-1] #可能是最愚蠢的错误，之前用了旋转矩阵

    return curve_data, end_point, matrix, s_phi

