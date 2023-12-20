import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# 输入音频文件和输出目录
audio_file = 'database/8m2d/小黑2.m4a'
output_dir = 'database/8m2d/'

# 创建输出目录
os.makedirs(output_dir + 'timed', exist_ok=True)
os.makedirs(output_dir + 'mfcc', exist_ok=True)
os.makedirs(output_dir + 'stft', exist_ok=True)

# 创建结果目录
result_dir = 'database/8m2d/result/'
os.makedirs(result_dir, exist_ok=True)

# 读取音频文件
y, sr = librosa.load(audio_file)

# 将音频切割成一秒一秒的片段
duration = 5  # 每个片段的时长（秒）
samples_per_segment = int(sr * duration)

for i, start_sample in enumerate(range(0, len(y), samples_per_segment)):
    segment = y[start_sample:start_sample + samples_per_segment]

    # 保存时域图像
    plt.figure(figsize=(5, 3))
    librosa.display.waveshow(segment, sr=sr)
    plt.title(f'Time Domain - Segment {i + 1}')
    plt.savefig(f'{output_dir}timed/segment_{i + 1}.png')
    plt.close()

    # 计算MFCC
    mfccs = librosa.feature.mfcc(y=segment, sr=sr)
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'MFCC - Segment {i + 1}')
    plt.savefig(f'{output_dir}mfcc/segment_{i + 1}.png')
    plt.close()

    # 计算STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(segment)))
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'STFT - Segment {i + 1}')
    plt.savefig(f'{output_dir}stft/segment_{i + 1}.png')
    plt.close()

    # 保存图像到结果目录，使用不同的文件名模式确保唯一性
    plt.figure(figsize=(5, 3))
    librosa.display.waveshow(segment, sr=sr)
    plt.title(f'Time Domain - Segment {i + 1}')
    plt.savefig(f'{result_dir}segment_{i + 1}_timed.png')
    plt.close()

    plt.figure(figsize=(5, 3))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'MFCC - Segment {i + 1}')
    plt.savefig(f'{result_dir}segment_{i + 1}_mfcc.png')
    plt.close()

    plt.figure(figsize=(5, 3))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f'STFT - Segment {i + 1}')
    plt.savefig(f'{result_dir}segment_{i + 1}_stft.png')
    plt.close()
