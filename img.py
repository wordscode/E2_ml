import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

data_root = "../17flowers/"

# 读取文件夹下的所有图像文件
image_files = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.endswith(".jpg")]

# 初始化三通道像素值列表
red_pixels = []
green_pixels = []
blue_pixels = []

# 循环读取每个图像并提取像素值
for image_file in image_files:
    # 读取图像
    image = cv2.imread(image_file)

    resized_image = cv2.resize(image, (200, 200))  # 调整为较小的尺寸

    # 提取红、绿、蓝通道像素值
    red_channel = resized_image[:, :, 2].flatten()
    green_channel = resized_image[:, :, 1].flatten()
    blue_channel = resized_image[:, :, 0].flatten()

    # 释放图像内存
    del image
    del resized_image

    # 将像素值添加到对应通道的列表中
    red_pixels.extend(red_channel)
    green_pixels.extend(green_channel)
    blue_pixels.extend(blue_channel)

# 绘制三通道像素值的直方图
plt.figure(figsize=(10, 6))

# 绘制红色通道直方图
plt.hist(red_pixels, bins=256, color='red', alpha=0.5, label='Red')

# 绘制绿色通道直方图
plt.hist(green_pixels, bins=256, color='green', alpha=0.5, label='Green')

# 绘制蓝色通道直方图
plt.hist(blue_pixels, bins=256, color='blue', alpha=0.5, label='Blue')

plt.xlabel('像素值')
plt.ylabel('频率')
plt.title('RGB像素值的直方图')
plt.legend()
plt.show()
