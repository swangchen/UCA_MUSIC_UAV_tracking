import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load and preprocess data
data_path = '定位数据库/states.csv'  # Update with your actual path
df = pd.read_csv(data_path)

# Select relevant features
features = ['lat', 'lon', 'geoaltitude']
data = df[features].dropna().values

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Split data into sequences
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append((seq, label))
    return sequences

seq_length = 5
sequences = create_sequences(data_normalized, seq_length)
X, y = zip(*sequences)
X = np.array(X)
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

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

    def forward(self, input_data, n_predictions):
        predictions = []
        for data in input_data:
            xt, vt, at = data
            xt = xt.to(self.embedding_layer.weight.device)
            vt = vt.to(self.embedding_layer.weight.device)
            at = at.to(self.embedding_layer.weight.device)

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

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        mse = nn.MSELoss()(y_pred, y_true)
        mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8)))
        return self.alpha * mse + (1 - self.alpha) * mape

input_size = 3
hidden_size = 128
output_size = 3
max_a = 0.5
dt = 0.1
embedded_size = 10
n_predictions = 3

uav_model = UAVModel(input_size, hidden_size, output_size, max_a, dt, embedded_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uav_model.to(device)

criterion = CustomLoss(alpha=0.5)
optimizer = optim.Adam(uav_model.parameters(), lr=0.01)
epochs = 10

# Train the model
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(len(X_train)):
        data = (X_train[i].unsqueeze(0), X_train[i].unsqueeze(0), X_train[i].unsqueeze(0))  # Mock data
        optimizer.zero_grad()
        output_data = uav_model([data], n_predictions)
        loss = 0
        for gru_x_outputs, gru_v_outputs, gru_a_outputs in output_data:
            loss += criterion(gru_x_outputs, y_train[i].unsqueeze(0))
            loss += criterion(gru_v_outputs, y_train[i].unsqueeze(0))  # Mock target
            loss += criterion(gru_a_outputs, y_train[i].unsqueeze(0))  # Mock target
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(X_train)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

# Plot the loss curve
plt.plot(losses, label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('loss_curve.png')

# Evaluate the model
uav_model.eval()
predictions = []
with torch.no_grad():
    for i in range(len(X_test)):
        data = (X_test[i].unsqueeze(0), X_test[i].unsqueeze(0), X_test[i].unsqueeze(0))  # Mock data
        output_data = uav_model([data], n_predictions)
        gru_x_outputs, gru_v_outputs, gru_a_outputs = output_data[0]
        predictions.append(gru_x_outputs[-1].cpu().numpy())

predictions = np.array(predictions)
y_test = y_test.cpu().numpy()

# Calculate evaluation metrics
mse = np.mean((predictions - y_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions - y_test))
mape = np.mean(np.abs((predictions - y_test) / (y_test + 1e-8)))

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")

# Plot the 3D trajectory for a subset of test data
subset_idx = np.random.choice(len(y_test), size=100, replace=False)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_test[subset_idx, 0], y_test[subset_idx, 1], y_test[subset_idx, 2], c='b', label='True')
ax.scatter(predictions[subset_idx, 0], predictions[subset_idx, 1], predictions[subset_idx, 2], c='r', label='Predicted')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Altitude')
plt.legend()
plt.savefig('trajectory_comparison.png')
