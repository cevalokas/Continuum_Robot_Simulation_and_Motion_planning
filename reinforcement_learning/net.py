import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义神经网络模型
class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 18),
            nn.ELU(),
            nn.Linear(18, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.ELU(64, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.fc(x)
    
# 比较花哨的
class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 18),
            nn.ReLU(),
            nn.BatchNorm1d(18),
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        return self.fc(x)

#比较简单的
class Net_3(nn.Module):
    def __init__(self):
        super(Net_3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.fc(x)

#批量规范化
class Net_4(nn.Module):
    def __init__(self):
        super(Net_4, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),  # 加入批量规范化
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.fc(x)
    
#在4的基础上增加了dropout，减少了层数
class Net_5(nn.Module):
    def __init__(self):
        super(Net_5, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),  # 加入批量规范化
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout层，丢弃概率为0.5
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.fc(x)

# 相比4，删掉最大一层参数
class Net_6(nn.Module):
    def __init__(self):
        super(Net_6, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),  # 加入批量规范化
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),  # 加入批量规范化
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        return self.fc(x)
    

# 为更严格的控制准备，删掉两个无关紧要的phi
class Net_7(nn.Module):
    def __init__(self):
        super(Net_7, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 12),
            nn.ELU(),
            nn.Linear(12, 24),
            nn.ELU(),
            nn.Linear(24, 16),
            nn.ELU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.fc(x)
    
# 干脆不使用网络推导phi，简化为二维问题，然后就不是过驱动了
class Net_8(nn.Module):
    def __init__(self):
        super(Net_8, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 8),
            nn.ELU(),
            nn.Linear(8, 24),
            nn.ELU(),
            nn.Linear(24, 12),
            nn.ELU(),
            nn.Linear(12, 3)
        )

    def forward(self, x):
        return self.fc(x)
    

class Net2D(nn.Module):
    def __init__(self):
        super(Net2D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 6),
            nn.Sigmoid(),
            nn.Linear(6, 18),
            nn.Sigmoid(),
            nn.Linear(18, 3)
        )

    def forward(self, x):
        return self.fc(x)