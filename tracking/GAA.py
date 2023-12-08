import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class UAVModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_a, dt, embedded_size):
        super(UAVModel, self).__init__()
        self.embedding_layer = nn.Linear(input_size, embedded_size)
        self.gru_x = nn.GRU(embedded_size, hidden_size)
        self.gru_v = nn.GRU(embedded_size, hidden_size)
        self.gru_a = nn.GRU(embedded_size, hidden_size)
        self.max_a = max_a
        self.dt = dt
        self.output_size = output_size

        self.attn_engine = nn.Sequential(
            nn.Linear(21, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )

    def attn(self, xo, vo, ao):
        inp = torch.cat((xo, vo, ao), dim=1)
        out = self.attn_engine(inp)
        w = F.softmax(out, dim=1)
        w1, w2, w3, w4 = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
        return w1, w2, w3, w4

    def moderate(self, w1, w2, w3, w4, xo, vo, at):
        xt = w3 * xo + w4 * (xo + vo)
        vt = w1 * vo + w2 * (vo + at)
        at = vo - vo
        return xt, vt, at

    # 在 forward 方法中确保数据在相同设备上
    def forward(self, input_data, n_predictions):
        predictions = []
        for data in input_data:
            xt, vt, at = data
            xt = xt.to(self.embedding_layer.weight.device)  # 将数据移动到与模型相同的设备
            vt = vt.to(self.embedding_layer.weight.device)  # 将数据移动到与模型相同的设备
            at = at.to(self.embedding_layer.weight.device)  # 将数据移动到与模型相同的设备

            gru_x_outputs, gru_v_outputs, gru_a_outputs = [], [], []
            for _ in range(n_predictions):
                w1, w2, w3, w4 = self.attn(xt, vt, at)
                xt_new, vt_new, at_new = self.moderate(w1, w2, w3, w4, xt, vt, at)

                xt_emb = self.embedding_layer(xt_new).unsqueeze(0)
                vt_emb = self.embedding_layer(vt_new).unsqueeze(0)
                at_emb = self.embedding_layer(at_new).unsqueeze(0)

                gru_x_output, _ = self.gru_x(xt_emb)
                gru_v_output, _ = self.gru_v(vt_emb)
                gru_a_output, _ = self.gru_a(at_emb)

                gru_x_outputs.append(gru_x_output.squeeze(0).squeeze(0))
                gru_v_outputs.append(gru_v_output.squeeze(0).squeeze(0))
                gru_a_outputs.append(gru_a_output.squeeze(0).squeeze(0))

                xt, vt, at = xt_new, vt_new, at_new

            gru_x_outputs = torch.stack(gru_x_outputs, dim=0)
            gru_v_outputs = torch.stack(gru_v_outputs, dim=0)
            gru_a_outputs = torch.stack(gru_a_outputs, dim=0)

            predictions.append((gru_x_outputs, gru_v_outputs, gru_a_outputs))

        return predictions


# 重构模型
input_size = 7
hidden_size = 128
output_size = 7
max_a = 0.5
dt = 0.1
embedded_size = 10
n_predictions = 3

uav_model = UAVModel(input_size, hidden_size, output_size, max_a, dt, embedded_size)


# 生成数据集
def generate_data_set(input_size, data_size):
    data = []
    for i in range(1, data_size * input_size + 1, input_size):
        x = torch.tensor([list(range(i, i + input_size))], dtype=torch.float32)
        v = 4 * x * x - 3
        a = 6 * x * x - 2 * x
        data.append((x, v, a))
    return data


data_size = 100
input_data = generate_data_set(input_size, data_size)
# input_data=np.array(input_data)
# print(input_data.shape)

# 定义训练参数
criterion = nn.MSELoss()
optimizer = optim.Adam(uav_model.parameters(), lr=0.01)
epochs = 10

# 将模型和数据移动到CUDA设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uav_model.to(device)

# 训练循环
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    for data in input_data:
        optimizer.zero_grad()
        output_data = uav_model([data], n_predictions)
        loss = 0
        for gru_x_outputs, gru_v_outputs, gru_a_outputs in output_data:
            loss += criterion(gru_x_outputs, torch.zeros_like(gru_x_outputs))
            loss += criterion(gru_v_outputs, torch.zeros_like(gru_v_outputs))
            loss += criterion(gru_a_outputs, torch.zeros_like(gru_a_outputs))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / data_size
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

# 画出损失函数图表
plt.plot(losses, label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')

# 保存模型
torch.save(uav_model.state_dict(), 'uav_model.pth')
print('模型保存成功')
