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
        print("噪声空间 UN形状：")
        print(noise_space.shape)

        kelm = 16  # X轴Y轴各自的阵元数量
        r = 0.018  # 圆阵半径

        # 假设已经有了UN
        UN = noise_space
        UN = UN.T  # 转置使得UN的shape为(16, 13)

        P_MUSIC_c = np.zeros((360, 180), dtype=np.complex_)  # 用于存储P_MUSIC_c的数组
        SP_C = np.zeros((360, 180), dtype=np.complex_)  # 用于存储SP_C的数组

        for ang1 in range(1, 361):
            for ang2 in range(1, 181):
                a = np.exp(-1j * 2 * np.pi * r * np.sin(ang2 * np.pi / 180) * np.cos(
                    ang1 * np.pi / 180 - 2 * np.pi * np.arange(kelm) / kelm))
                a = a[:, np.newaxis]  # 将a转换为形状为(16, 1)的列向量
                SP_C[ang1 - 1, ang2 - 1] = np.dot(a.conj().T, np.dot(UN, UN.conj().T)).dot(a.conj())

        # 找到最小的 channel_count 个值
        min_indices_SP_C = np.argpartition(SP_C, channel_count, axis=None)[:channel_count]
        min_values_SP_C = SP_C.flat[min_indices_SP_C]

        print(f"SP_C的最小 {channel_count} 个值为：{min_values_SP_C}")
        min_indices_SP_C = np.unravel_index(min_indices_SP_C, SP_C.shape)
        print("最小值出现在以下位置：")
        for i in range(channel_count):
            print(f"ang1 = {min_indices_SP_C[0][i] + 1}，ang2 = {min_indices_SP_C[1][i] + 1}")

    print(f"已完成 {processed_channels + 16} 个通道的处理。")

print("循环完成。")
