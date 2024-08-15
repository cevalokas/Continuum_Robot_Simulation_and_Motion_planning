import numpy as np
from Kinematics.kine2D import kine2D

class RLModel:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay

        # 动作空间：theta的增量
        self.actions = np.linspace(-0.1, 0.1, 21)  # 21个动作，从-0.1到0.1

        # 初始化 Q 表
        self.q_table = {}
        self.init_q_table()

    def init_q_table(self):
        theta_range = np.linspace(-np.pi, np.pi, 360)
        for theta1 in theta_range:
            for theta2 in theta_range:
                for theta3 in theta_range:
                    self.q_table[(theta1, theta2, theta3)] = np.zeros(len(self.actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.actions))
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train(self):
        current_state = tuple(self.env.theta)
        action_idx = self.choose_action(current_state)
        action = self.actions[action_idx]

        # 应用动作
        self.env.theta += action

        # 计算reward：目标是最小化曲线末端和目标点之间的距离
        curves = kine2D(self.env.theta)
        end_point = np.array([curves[-1][0][-1], curves[-1][1][-1]])
        distance_to_target = np.linalg.norm(end_point - self.env.target)
        reward = -distance_to_target

        next_state = tuple(self.env.theta)
        self.update_q_table(current_state, action_idx, reward, next_state)

        # 衰减探索率
        self.epsilon *= self.epsilon_decay

def RL_train(env):
    model = RLModel(env)
    model.train()
