import numpy as np  # 导入NumPy库，用于科学计算
import scipy.signal as ss  # 导入SciPy中的信号处理模块
import scipy.linalg as LA  # 导入SciPy中的线性代数模块
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图

derad = np.pi / 180  # 弧度转换为角度的系数
radeg = 180 / np.pi  # 角度转换为弧度的系数

# 添加高斯白噪声
def awgn(x, snr):
    spower = np.sum((np.abs(x) ** 2)) / x.size
    x = x + np.sqrt(spower / snr) * (np.random.randn(x.shape[0], x.shape[1]) + 1j * np.random.randn(x.shape[0], x.shape[1]))
    return x

# MUSIC算法的实现
def MUSIC(K, d, theta, snr, n):
    iwave = theta.size  # 信号波束的数量
    tao = d.reshape(-1, 1) @ np.sin(theta * derad)#d sin（theta）
    A = np.exp(-1j * 2 * np.pi * tao)  # 传感器的响应矩阵
    S = np.random.randn(iwave, n)  # 生成随机信号
    X = A @ S  # 生成接收信号，矩阵乘法
    X = awgn(X, 10)  # 添加高斯白噪声
    Rxx = X @ (X.conj().T) / n  # 估计信号的协方差矩阵
    D, EV = LA.eig(Rxx)  # 对协方差矩阵进行特征值分解
    index = np.argsort(D)  # 对特征值进行排序
    EN = EV.T[index].T[:, 0:K - iwave]  # 选取特征值最小的K-iwave列

    SP = np.empty(numAngles, dtype=complex)  # 用于存储MUSIC算法的空间谱

    for i in range(numAngles):
        a = np.exp(-1j * 2 * np.pi * d.reshape(-1, 1) * np.sin(Angles[i]))  # 信号波束的方向向量
        SP[i] = ((a.conj().T @ a) / (a.conj().T @ EN @ EN.conj().T @ a))[0, 0]  # MUSIC算法的空间谱

    return SP

# 角度范围
Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)  # 角度范围数组
numAngles = Angles.size  # 角度范围数组的大小

# 用于MUSIC算法的参数设置
d = np.arange(0, 4, 0.5)  # 传感器之间的间距数组
# 信号到达角度数组
theta = np.array([10, 30, 60]).reshape(1, -1)
SP = np.empty(numAngles, dtype=complex)  # 用于存储MUSIC算法的空间谱
SP = MUSIC(K=8, d=d, theta=theta, snr=10, n=500)  # 使用MUSIC算法进行信号角度估计

# 对结果进行处理并绘制图像
SP = np.abs(SP)  # 计算空间谱的模
SPmax = np.max(SP)  # 空间谱的最大值
SP = 10 * np.log10(SP / SPmax)  # 将空间谱转换为分贝单位
x = Angles * radeg  # 将角度转换为角度单位
plt.plot(x, SP)  # 绘制空间谱图像
plt.show()  # 显示图像
