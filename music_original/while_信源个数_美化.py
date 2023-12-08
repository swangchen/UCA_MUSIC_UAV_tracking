import numpy as np
import matplotlib.pyplot as plt

# 定义总的通道数和循环次数
total_channels = 16
iterations = 1000000
data_points = 3000

for iteration in range(iterations):
    print(f"这是第{iteration + 1}个循环。")
    for processed_channels in range(0, total_channels, 16):
        # 创建一个16x3000的零矩阵
        matrix = np.zeros((16, data_points))

        for i in range(processed_channels, processed_channels + 16):
            file_path = f"audiov2/audio_opt{i}.raw"
            with open(file_path, "rb") as f:  # 以二进制模式打开文件
                content = f.read(2 * data_points)  # 读取每个文件的前3000个数值
                content = np.frombuffer(content, dtype=np.int16)  # 以int16格式解析二进制数据
                matrix[i - processed_channels, :] = content  # 将数值放入矩阵中

        # 求自相关矩阵和自相关矩阵特征值
        autocorr_matrix = np.corrcoef(matrix)

        # 计算特征值
        eigenvalues = np.linalg.eigvals(autocorr_matrix)

        # 找到小于最大特征值的三分之一的特征值
        max_eigenvalue = max(eigenvalues)
        threshold = max_eigenvalue / 10
        selected_eigenvalues = eigenvalues[eigenvalues < threshold]

        # 计算特征根的数量
        count = len(selected_eigenvalues)
        channel_count = 16 - count
        print(f"信源数量为{channel_count}")

print("循环完成。")
