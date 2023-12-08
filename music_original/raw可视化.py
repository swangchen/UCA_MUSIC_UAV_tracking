import numpy as np
import matplotlib.pyplot as plt

# 读取音频文件并可视化
for i in range(16):
    file_name = f'audiov2/audio_opt{i}.raw'
    with open(file_name, 'rb') as f:
        audio = np.fromfile(f, dtype=np.int16)

    # 创建时间轴
    time = np.linspace(0, len(audio) / 44100, num=len(audio))

    # 绘制波形图
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio)
    plt.title(f'Audio Opt{i} Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
