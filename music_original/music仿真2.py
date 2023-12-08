import numpy as np
import scipy.signal as ss
import scipy.linalg as LA
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 将角度和弧度互相转换的常数
derad = np.pi / 180
radeg = 180 / np.pi

# 加性高斯白噪声函数
def awgn(x, snr):
    spower = np.sum((np.abs(x) ** 2)) / x.size
    x = x + np.sqrt(spower / snr) * (np.random.randn(x.shape[0], x.shape[1]) + 1j * np.random.randn(x.shape[0], x.shape[1]))
    return x

# MUSIC算法实现函数
def MUSIC(K, pos, theta, snr, n):
    # 计算参数初始化
    iwave = theta.size
    d = 0.03534  # 修改为周长的16分之一
    A = np.exp(-1j * 2 * np.pi * d * np.sin(theta * derad))
    S = np.random.randn(iwave, n)
    X = A @ S
    X = awgn(X, 10)
    Rxx = X.dot(X.conj().T) / n
    D, EV = LA.eig(Rxx)
    index = np.argsort(D)
    EN = EV[:, index[0:K - iwave]]

    SP = np.empty(numAngles, dtype=complex)

    # 计算空间谱
    for i in range(numAngles):
        d = 0.03534  # 修改为周长的16分之一
        a = np.exp(-1j * 2 * np.pi * d * np.sin(Angles[i]))
        SP[i] = ((np.conj(a).T.dot(a)) / (np.conj(a).T.dot(EN).dot(EN.conj().T).dot(a)))[0]

    return SP

# 设定角度范围
Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
numAngles = Angles.size

# 计算麦克风位置
d = 0.03534
num_microphones = 16
pos = []
for i in range(num_microphones):
    angle = 2 * np.pi * i / num_microphones
    x = d * np.cos(angle)
    y = d * np.sin(angle)
    pos.append([x, y, 0])
pos = np.array(pos)

# 设置参数并运行MUSIC算法
theta = np.array([10, 30, 60]).reshape(1, -1)
SP = np.empty(numAngles, dtype=complex)
SP = MUSIC(K=16, pos=pos, theta=theta, snr=1, n=500)

# 处理结果并绘制图形
SP = np.abs(SP)
SPmax = np.max(SP)
SP = 10 * np.log10(SP / SPmax)
x = Angles * radeg
plt.plot(x, SP)
plt.title('MUSIC算法空间谱')
plt.xlabel('角度 (度)')
plt.ylabel('空间谱 (dB)')
plt.show()
