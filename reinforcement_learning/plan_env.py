import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
from Kinematics.kine2D import kine2D
from reinforcement_learning.RL_train import RL_train

class Env:
    def __init__(self):
        # 创建图形和轴
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-2, 6)

        # 初始化三段曲线，使用不同的颜色和粗细
        self.lines = [self.ax.plot([], [], lw=2, color='r')[0],
                      self.ax.plot([], [], lw=2, color='g')[0],
                      self.ax.plot([], [], lw=2, color='b')[0]]

        # 初始化 theta
        self.theta = np.zeros(3)
        # self.theta = np.array([-0.8942,  1.0617,  0.1831])
        # 随机一个目标位置
        self.target = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-1, 2.5)
        ])
        # self.target = np.array([-2.2528,  0.5652])

        # 在图形上添加目标点
        self.target_point, = self.ax.plot(self.target[0], self.target[1], 'yo', markersize=10)

        # 全局变量来跟踪动画是否暂停
        self.is_paused = False

        # 监听键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # 创建动画，不设置frames使其无限循环
        self.ani = FuncAnimation(self.fig, self.update, blit=True)

    def update(self, frame):
        if not self.is_paused:
            curves = kine2D(self.theta)
            RL_train(self)  # 作用是随更新微调theta，使每次update时curves的末端逐渐接近target
            for i, line in enumerate(self.lines):
                line.set_data(curves[i][0], curves[i][1])
            time.sleep(0.1)
        return self.lines + [self.target_point]

    def on_key(self, event):
        if event.key == 'p':
            self.is_paused = not self.is_paused
        elif event.key == 'escape':
            plt.close()  # 关闭图形窗口以停止动画

    def show(self):
        plt.show()

# 实例化Env类并显示动画
env = Env()
env.show()
