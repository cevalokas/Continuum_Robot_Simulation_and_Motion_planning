import numpy as np
import math

def kine(pose, length=1):
    # end_point初始化为原点
    end_point = np.array([0.0, 0.0, 0.0])
    # 初始化旋转矩阵为单位矩阵
    matrix = np.eye(3)
    s_phi = 0
    for j in range(3):
        phi = pose[j]
        theta = pose[j + 3]
        loc = np.array([0.0, 0.0, 0.0])
        
        if theta != 0:
            loc[2] = math.sin(length * theta) / theta
            c = (1 - math.cos(length * theta)) / theta
            loc[0] = c * math.cos(phi)
            loc[1] = c * math.sin(phi)
        else:
            loc[2] = length
        
        # 使用旋转矩阵将当前loc从局部坐标系变换到全局坐标系
        loc = np.dot(matrix, loc)
        # 累加end_point以得到全局坐标
        loc += end_point
        # print(f"loc{j}={loc}")
        
        # 计算nloc（当前段缩短0.01长度后的新位置）
        nloc = np.array([0.0, 0.0, 0.0])
        nlength = length - 0.01
        
        if theta != 0:
            nloc[2] = math.sin(nlength * theta) / theta
            c = (1 - math.cos(nlength * theta)) / theta
            nloc[0] = c * math.cos(phi)
            nloc[1] = c * math.sin(phi)
        else:
            nloc[2] = nlength
        
        nloc = np.dot(matrix, nloc)
        nloc += end_point
        
        # 计算曲线末端的切向量
        w = loc - nloc
        w /= np.linalg.norm(w)  # 归一化
        
        s_phi+=phi
        # 计算新的旋转矩阵的列向量
        z = -w[0] * math.cos(s_phi) - w[1] * math.sin(s_phi)
        v = np.array([math.cos(s_phi), math.sin(s_phi), z])
        v /= np.linalg.norm(v)
        u = np.cross(w, v)
        u /= np.linalg.norm(u)
        matrix = np.column_stack((v, u, w))
        
        # 更新end_point
        end_point = loc
    
    return loc

if __name__ == "__main__":
    pose = np.array([0, 0, 0, 0, 0, 0])
    kine(pose)
