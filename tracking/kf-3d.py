import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据并预处理
df = pd.read_csv('定位数据库/states.csv')

# 选择需要的特征，并删除包含NaN的行
df = df[['time', 'lat', 'lon', 'geoaltitude']].dropna()

# 仅使用部分数据
df = df.iloc[:1000]  # 取前1000行数据

# 标准化特征
features = ['lat', 'lon', 'geoaltitude']
df[features] = (df[features] - df[features].mean()) / df[features].std()

# 将数据转换为 numpy 数组
data = df[features].values

# 定义Kalman Filter
class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # 状态转移矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 测量噪声协方差
        self.P = P  # 估计误差协方差
        self.x = x  # 状态向量

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return self.x

# 初始化Kalman Filter参数
dt = 1  # 时间间隔
F = np.array([[1, dt, 0.5*dt**2, 0, 0, 0],
              [0, 1, dt, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5*dt**2],
              [0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1]])
Q = np.eye(6) * 0.001  # 过程噪声协方差
R = np.eye(3) * 0.1    # 测量噪声协方差
P = np.eye(6)          # 估计误差协方差
x = np.zeros(6)        # 初始状态向量

kf = KalmanFilter(F, H, Q, R, P, x)

# 预测和更新状态
predictions = []
for z in data:
    kf.predict()
    x = kf.update(z)
    predictions.append(x)

predictions = np.array(predictions)

# 反标准化特征
df[features] = df[features] * df[features].std() + df[features].mean()
predictions[:, [0, 3, 5]] = predictions[:, [0, 3, 5]] * df[features].std().values + df[features].mean().values

# 计算误差
rmse = np.sqrt(np.mean((data - predictions[:, [0, 3, 5]]) ** 2))
mae = np.mean(np.abs(data - predictions[:, [0, 3, 5]]))
mape = np.mean(np.abs((data - predictions[:, [0, 3, 5]]) / data)) * 100

print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')

# 选择连续的一段数据进行绘图
n_points = 50  # 选择连续的50个点
start_idx = 100  # 从第100个点开始
end_idx = start_idx + n_points

true_trajectory = data[start_idx:end_idx]
predicted_trajectory = predictions[start_idx:end_idx, [0, 3, 5]]

# 绘制3D预测结果
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], label='True Trajectory', color='b')
ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], label='Predicted Trajectory', color='r', linestyle='--')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Geoaltitude')
ax.set_title('3D Trajectory Prediction with Kalman Filter')
ax.legend()

plt.savefig('kf_3d_trajectory.png')
plt.show()
