import os
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 没有直接移动图像文件到不同的文件夹，而是将图像的路径存储到不同的列表中。
def read_split_data(root: str, test_rate: float = 0.2):
    random.seed(0)  # 保证随机划分结果一致
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".jpeg", ".JPEG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样测试样本
        test_path = random.sample(images, k=int(len(images) * test_rate))

        for img_path in images:
            if img_path in test_path:  # 如果该路径在采样的测试集样本中则存入测试集
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.\n{} for training, {} for testing".format(sum(every_class_num),
                                                                                         len(train_images_path),
                                                                                         len(test_images_path)
                                                                                         ))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, test_images_path, test_images_label


def generate_ds(data_root: str,
                train_im_height: int = None,
                train_im_width: int = None,
                test_im_height: int = None,
                test_im_width: int = None,
                batch_size: int = 8,
                test_rate: float = 0.2,
                cache_data: bool = False):
    """
        读取划分数据集，并生成训练集和验证集的迭代器
        :param data_root: 数据根目录
        :param train_im_height: 训练输入网络图像的高度
        :param train_im_width:  训练输入网络图像的宽度
        :param test_im_height: 测试输入网络图像的高度
        :param test_im_width:  测试输入网络图像的宽度
        :param batch_size: 训练使用的batch size
        :param test_rate:  将数据按给定比例划分到测试集
        :param cache_data: 是否缓存数据
        :return:
    """
    assert train_im_height is not None
    assert train_im_width is not None
    if test_im_width is None:
        test_im_width = train_im_width
    if test_im_height is None:
        test_im_height = train_im_height

    train_img_path, train_img_label, test_img_path, test_img_label = read_split_data(data_root, test_rate=test_rate)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # 图像数据进行预处理
    def process_train_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, train_im_height, train_im_width)
        image = tf.image.random_flip_left_right(image)
        image = (image / 255. - 0.5) / 0.5
        return image, label

    def process_test_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, test_im_height, test_im_width)
        image = (image / 255. - 0.5) / 0.5
        return image, label

    # 配置数据集以提高性能
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)  # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)  # 在训练的同时提前准备下一个step的数据
        return ds
    # 只有在迭代访问数据时才会逐批次地加载图像数据
    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_path),
                                                   tf.constant(train_img_label)))
    total_train = len(train_img_path)  # 训练集中的图像数量
    # 使用Dataset.map创建一个图像、标签对的数据集
    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)
    # tf.constant函数的默认数据类型是int32,导致测试标签...
    test_ds = tf.data.Dataset.from_tensor_slices((tf.constant(test_img_path),
                                                  tf.constant(test_img_label)))
    total_test = len(test_img_path)
    # 使用Dataset.map创建一个图像、标签对的数据集
    test_ds = test_ds.map(process_test_info, num_parallel_calls=AUTOTUNE)
    test_ds = configure_for_performance(test_ds, total_test, cache=False)

    return train_ds, test_ds
