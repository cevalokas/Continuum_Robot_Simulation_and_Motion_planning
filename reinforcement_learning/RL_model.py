import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Kinematics.kine import kine
from reinforcement_learning.net import Net_7

class RL_model:
    def __init__(self, APP):
        self.app = APP
        self.model_path = './model_libs/Net7_2_model.pth'
        self.model = Net_7()

    def load_data(self):
        target_params = {k: v.get() for k, v in self.app.target_entries.items()}
        target_pos = torch.tensor([[float(target_params['x']), float(target_params['y']), float(target_params['z'])]])  # 新的末端位置
        return target_pos

    def impossible(self,data) -> bool:
        if np.linalg.norm(data) > 3*int(self.app.length_entry.get()):
            return True
        else:
            return False

    def run(self):
        data = self.load_data()
        if (self.impossible(data)):
            return print("It is impossible!")
        # 挑选并加载使用模型
        self.model.load_state_dict(torch.load(self.model_path))

        with torch.no_grad():
            p_posture = self.model(data)
        p_posture = p_posture.tolist()[0]
        
        # 要插入补全的位置和元素
        index_to_insert = 1
        values_to_insert = [0, 0]
        # 插入元素
        f_posture = np.insert(p_posture, index_to_insert, values_to_insert)

        keys = ['phi0','phi1','phi2','theta0','theta1', 'theta2']
        p = dict(zip(keys, f_posture))

        for key, value in p.items():
            if key in self.app.posture_entries:
                entry = self.app.posture_entries[key]
                entry.delete(0, 'end')  # 删除当前的值
                entry.insert(0, str(value))  # 插入新的值
            else:
                print(f"No entry found for key: {key}")

        self.app.update_plot()
