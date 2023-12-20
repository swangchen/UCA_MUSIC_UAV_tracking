# 指定输出文件路径
output_file = 'output.txt'
import random
# 打开输出文件以写入数据
with open('output.txt', 'w') as f:
    counter = 1
    for i in range(258, 291, 3):
        if i <= 81:
            line = f"image_{i},image_{i+1},image_{i+2},1,0\n"
        else:
            line = f"image_{i},image_{i+1},image_{i+2},2,0\n"
        f.write(line)


print(f"Generated {output_file}")
