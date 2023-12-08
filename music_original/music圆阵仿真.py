import numpy as np
import matplotlib.pyplot as plt

# 设置初始参数
derad = np.pi / 180
N = 8  # 阵元个数
M = 3  # 信源数目
theta = np.array([-50, 0, 50])  # 待估计角度
snr = 10  # 信噪比
K = 1024  # 快拍数 10/1024

dd = 0.5  # 阵元间距 d=lamda/2
d = np.arange(0, N) * dd
A = np.exp(-1j * 2 * np.pi * np.outer(d, np.sin(theta * derad)))

S = np.random.randn(M, K)
X = A.dot(S)
X1 = X + np.random.normal(0, 1, size=X.shape) * np.sqrt(np.mean(np.abs(X) ** 2) / (10 ** (snr / 10)))

Rxx = X1.dot(X1.T) / K
EV, D = np.linalg.eig(Rxx)

EVA = np.sort(np.diag(D))
EV = EV[:, np.argsort(np.diag(D))]  # 更正为 EV = EV[:, np.argsort(np.diag(D))]

SP = np.zeros(361, dtype=complex)
angles = np.linspace(-100, 100, 361)

for iang in range(361):
    phim = derad * angles[iang]
    a = np.exp(-1j * 2 * np.pi * d * np.sin(phim))
    L = M
    En = EV[:, L:N]
    SP[iang] = 1 / (np.conj(a).T @ En @ En.conj().T @ a)

SP = np.abs(SP)
SP = 10 * np.log10(SP)
plt.plot(angles, SP, linewidth=0.5)
plt.xlabel('入射角/(degree)')
plt.ylabel('空间谱/(dB)')
plt.grid(True)
plt.show()
