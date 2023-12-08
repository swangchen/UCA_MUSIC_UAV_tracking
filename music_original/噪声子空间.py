import numpy as np

# 定义总的通道数和循环次数
total_channels = 16
iterations = 1
data_points = 300000

for _ in range(iterations):
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

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(autocorr_matrix)

        print("特征值为：", eigenvalues)

        # 画出特征值的散点图
        # ... （省略散点图的代码）

        # 找到小于最大特征值的三分之一的特征值
        max_eigenvalue = max(eigenvalues)
        threshold = max_eigenvalue / 10
        selected_eigenvalues = eigenvalues[eigenvalues < threshold]

        # 计算特征根的数量
        count = len(selected_eigenvalues)
        channel_count = 16 - count
        print(f"在聚类内的特征根数量为：{count}")
        print(f"信源数量为{channel_count}")

        # 构建噪声空间UN
        noise_space = []
        for i in range(len(eigenvalues)):
            if eigenvalues[i] < threshold:
                noise_space.append(eigenvectors[:, i])

        noise_space = np.array(noise_space)
        print("噪声空间 UN：")
        print(noise_space)
        print(noise_space.shape)

    print(f"已完成 {processed_channels + 16} 个通道的处理。")

print("循环完成。")
