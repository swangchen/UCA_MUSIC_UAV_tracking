import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# 将数据转换为PyTorch张量
data = torch.tensor(df[features].values, dtype=torch.float32)

# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 模型参数
input_size = len(features)
hidden_size = 128
output_size = len(features)
num_layers = 2
num_epochs = 100
learning_rate = 0.001

# 生成数据集
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return torch.stack(sequences), torch.stack(labels)

seq_length = 10
X, y = create_sequences(data, seq_length)

# 训练和测试集划分
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型、损失函数和优化器
model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 画出损失函数图表
plt.plot(losses, label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('gru_loss_curve.png')
plt.show()

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()

# 计算误差
rmse = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))
mae = mean_absolute_error(y_test.numpy(), y_pred)
mape = np.mean(np.abs((y_test.numpy() - y_pred) / y_test.numpy())) * 100

print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')

# 选择连续的一段数据进行绘图
n_points = 200  # 选择连续的50个点
start_idx = 100  # 从第100个点开始
end_idx = start_idx + n_points

true_trajectory = y_test[start_idx:end_idx].numpy()
predicted_trajectory = y_pred[start_idx:end_idx]

# 绘制3D预测结果
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], label='True Trajectory', color='b')
ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], label='Predicted Trajectory', color='r', linestyle='--')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Geoaltitude')
ax.set_title('3D Trajectory Prediction')
ax.legend()

plt.savefig('gru_3d_trajectory.png')
plt.show()
