"""
official code:
https://github.com/google/automl/tree/master/efficientnetv2
EfficientNetV1中存在的问题
训练图像的尺寸很大时，训练速度非常慢
在网络浅层中使用Depthwise convolutions速度会很慢
同等的放大每个stage是次优的

解决办法
降低训练图像的尺寸
通过NAS搜索，将stage2,3,4都替换成Fused-MBConv结构
非均匀的缩放策略来缩放模型
"""

import itertools
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# 卷积层和全连接层的权重初始化器的配置参数。
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',  # VarianceScaling类作为初始化器
    'config': {
        'scale': 2.0,  # 标准差为2.0倍的缩放
        'mode': 'fan_out',  # 仅考虑输出通道的数量。
        'distribution': 'truncated_normal'  # 分布类型为截断正态分布
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,  # 权重范围的1/3倍的缩放
        'mode': 'fan_out',
        'distribution': 'uniform'  # 分布类型为均匀分布
    }
}


# Squeeze-and-Excitation(SE)模块的实现，用于增强模型的表示能力。
class SE(layers.Layer):
    def __init__(self,
                 se_filters: int,  # 压缩操作中卷积层的滤波器数量。
                 output_filters: int,  # 扩张操作中卷积层的滤波器数量。
                 name: str = None):  # SE模块的名称
        super(SE, self).__init__(name=name)
        # 压缩操作，使用一个1x1的卷积层对输入进行降维操作
        self.se_reduce = layers.Conv2D(filters=se_filters,  # 滤波器数量
                                       kernel_size=1,  # 卷积核大小
                                       strides=1,  # 卷积步长
                                       padding="same",  # 填充方式保持输入和输出形状相同。
                                       activation="swish",  # 激活函数使用Swish函数。
                                       use_bias=True,  # 使用偏置项。
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,  # 卷积核的初始化器。
                                       name="conv2d")  # 卷积层的名称。
        # 扩张操作，使用一个1x1的卷积层对输入进行增维操作
        self.se_expand = layers.Conv2D(filters=output_filters,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       activation="sigmoid",  # 激活函数使用Sigmoid函数。
                                       use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       name="conv2d_1")

    # 定义了SE模块的前向传播过程。
    def call(self, inputs, **kwargs):
        # Tensor: [N, H, W, C] -> [N, 1, 1, C]
        # inputs: 输入张量，形状为[N, H, W, C],全局平均池化操作：通过对输入在H和W维度上进行平均池化，将形状变为[N, 1, 1, C]。
        # 对输入进行全局平均池化，输出形状为[N, 1, 1, C]
        se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
        # 对压缩操作进行前向传播，输出形状为[N, 1, 1, se_filters]
        se_tensor = self.se_reduce(se_tensor)
        #  对扩张操作进行前向传播，输出形状为[N, 1, 1, output_filters]
        se_tensor = self.se_expand(se_tensor)
        # 将SE模块的输出与输入进行元素级乘法，输出形状与输入相同
        return se_tensor * inputs


# 包含了一系列卷积层和规范化层，用于构建主体部分。
class MBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,       # 卷积核大小
                 input_c: int,           # 输入通道数
                 out_c: int,             # 输出通道数
                 expand_ratio: int,      # expansion ratio，扩张因子
                 stride: int,            # 步长
                 se_ratio: float = 0.25, # SE 模块中Squeeze操作后降维的比例
                 drop_rate: float = 0.,  # 随机丢弃率
                 name: str = None):      # 层名
        super(MBConv, self).__init__(name=name)

        if stride not in [1, 2]:    # 步长只能为1或2
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)   # 是否存在shortcut分支，当步长为1且输入输出通道数相等时才有
        expanded_c = input_c * expand_ratio    # 扩张后的通道数

        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))   # 获取BatchNormalization层名的函数
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))            # 获取Conv2D层名的函数

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1    # EfficientNetV2中不允许存在expand_ratio=1的情况，因此必须大于1
        # Point-wise expansion，Point-wise升维
        self.expand_conv = layers.Conv2D(
            filters=expanded_c,
            kernel_size=1,          # 卷积核大小为1，即Point-wise卷积
            strides=1,
            padding="same",
            use_bias=False,
            name=get_conv_name())   # Point-wise卷积层
        self.norm0 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())   # BatchNormalization层
        self.act0 = layers.Activation("swish")  # Swish激活函数

        # Depth-wise convolution，深度可分离卷积
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,    # 卷积核大小
            strides=stride,             # 步长
            depthwise_initializer=CONV_KERNEL_INITIALIZER,  # 初始化器
            padding="same",
            use_bias=False,
            name="depthwise_conv2d")    # 深度可分离卷积层
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())      # BatchNormalization层
        self.act1 = layers.Activation("swish")   # Swish激活函数

        # SE，Squeeze-and-Excitation模块
        num_reduced_filters = max(1, int(input_c * se_ratio))   # Squeeze操作后的通道数
        self.se = SE(num_reduced_filters, expanded_c, name="se") # SE模块层

        # Point-wise linear projection，Point-wise降维
        self.project_conv = layers.Conv2D(
            filters=out_c,
            kernel_size=1,          # 卷积核大小为1，即Point-wise卷积
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,  # 初始化器
            padding="same",
            use_bias=False,
            name=get_conv_name())    # Point-wise卷积层
        self.norm2 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())      # BatchNormalization层

        self.drop_rate = drop_rate    # 随机丢弃率
        if self.has_shortcut and drop_rate > 0:
            # Stochastic Depth
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                            name="drop_path")

    def call(self, inputs, training=None):
        x = inputs

        x = self.expand_conv(x)  # Point-wise升维
        x = self.norm0(x, training=training)  # BN
        x = self.act0(x)  # Swish激活函数

        x = self.depthwise_conv(x)  # 深度可分离卷积
        x = self.norm1(x, training=training)  # BN
        x = self.act1(x)  # Swish激活函数

        x = self.se(x)  # SE模块

        x = self.project_conv(x)  # Point-wise降维
        x = self.norm2(x, training=training)  # BN

        if self.has_shortcut:  # 如果存在shortcut分支
            if self.drop_rate > 0:  # 随机丢弃率大于0
                x = self.drop_path(x, training=training)  # 随机深度

            x = tf.add(x, inputs)  # 加上输入特征

        return x  # 返回输出特征


# 优化版本的MBConv模块，通过合并部分卷积层来提高计算效率
class FusedMBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,  # 卷积核大小
                 input_c: int,  # 输入的通道数
                 out_c: int,  # 输出的通道数
                 expand_ratio: int,  # 扩张比例
                 stride: int,  # 步长
                 se_ratio: float,  # Squeeze-and-Excitation模块中的缩放比例
                 drop_rate: float = 0.,  # 随机失活率
                 name: str = None):  # 该层的名称
        super(FusedMBConv, self).__init__(name=name)
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")  # 如果步长不合法，抛出异常

        assert se_ratio == 0.  # 断言Squeeze-and-Excitation模块的缩放比例为0

        self.has_shortcut = (stride == 1 and input_c == out_c)  # 是否为残差连接
        self.has_expansion = expand_ratio != 1  # 是否有扩张操作
        expanded_c = input_c * expand_ratio  # 扩张后的通道数

        bid = itertools.count(0)  # 生成器，生成一个无限的整数序列
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))  # 获取BatchNormalization层的名称
        cid = itertools.count(0)  # 生成器，生成一个无限的整数序列
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))  # 获取卷积层的名称

        if expand_ratio != 1:
            self.expand_conv = layers.Conv2D(
                filters=expanded_c,  # 扩张后的通道数
                kernel_size=kernel_size,  # 卷积核大小
                strides=stride,  # 步长
                kernel_initializer=CONV_KERNEL_INITIALIZER,  # 卷积核初始化器
                padding="same",  # 填充方式
                use_bias=False,  # 是否使用偏置
                name=get_conv_name())  # 层的名称
            self.norm0 = layers.BatchNormalization(
                axis=-1,
                momentum=0.9,  # 动量
                epsilon=1e-3,  # 防止除0错误
                name=get_norm_name())  # 层的名称
            self.act0 = layers.Activation("swish")  # 激活函数

        self.project_conv = layers.Conv2D(
            filters=out_c,  # 输出的通道数
            kernel_size=1 if expand_ratio != 1 else kernel_size,  # 如果有扩张操作，则卷积核大小为1，否则为kernel_size
            strides=1 if expand_ratio != 1 else stride,  # 如果有扩张操作，则步长为1，否则为stride
            kernel_initializer=CONV_KERNEL_INITIALIZER,  # 卷积核初始化器
            padding="same",  # 填充方式
            use_bias=False,  # 是否使用偏置
            name=get_conv_name())  # 层的名称
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,  # 动量
            epsilon=1e-3,  # 防止除0错误
            name=get_norm_name())  # 层的名称

        if expand_ratio == 1:
            self.act1 = layers.Activation("swish")  # 激活函数

        self.drop_rate = drop_rate  # 随机失活率
        if self.has_shortcut and drop_rate > 0:
            # Stochastic Depth
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # 噪声形状，即二元随机失活掩码
                                            name="drop_path")  # 层的名称

    def call(self, inputs, training=None):
        x = inputs
        if self.has_expansion:
            x = self.expand_conv(x)
            x = self.norm0(x, training=training)  # BN操作
            x = self.act0(x)  # 激活函数

        x = self.project_conv(x)  # 卷积操作
        x = self.norm1(x, training=training)
        if self.has_expansion is False:
            x = self.act1(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)  # 随机失活操作

            x = tf.add(x, inputs)  # 残差连接

        return x


# Stage0，用于对输入图像进行特征提取。
class Stem(layers.Layer):
    def __init__(self, filters: int, name: str = None):  # 卷积核数量和层名称
        super(Stem, self).__init__(name=name)
        self.conv_stem = layers.Conv2D(
            filters=filters,  # 卷积核数量
            kernel_size=3,  # 卷积核大小
            strides=2,  # 步长
            kernel_initializer=CONV_KERNEL_INITIALIZER,  # 卷积核初始化器
            padding="same",  # 填充方式
            use_bias=False,  # 是否使用偏置
            name="conv2d")  # 层的名称
        self.norm = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,  # 动量
            epsilon=1e-3,  # 防止除0错误
            name="batch_normalization")  # 层的名称
        self.act = layers.Activation("swish")  # 激活函数

    def call(self, inputs, training=None):
        x = self.conv_stem(inputs)  # 卷积操作
        x = self.norm(x, training=training)  # BN操作
        x = self.act(x)  # 激活函数

        return x  # 返回输出特征图


# Stage7，包含全局平均池化和全连接层，用于进行分类预测。
class Head(layers.Layer):
    def __init__(self,
                 filters: int = 1280,  # 卷积核数量
                 num_classes: int = 1000,  # 类别数
                 drop_rate: float = 0.,  # 随机失活率
                 name: str = None):  # 该层的名称
        super(Head, self).__init__(name=name)
        self.conv_head = layers.Conv2D(
            filters=filters,
            kernel_size=1,  # 卷积核大小为1
            kernel_initializer=CONV_KERNEL_INITIALIZER,  # 卷积核初始化器
            padding="same",  # 填充方式
            use_bias=False,  # 是否使用偏置
            name="conv2d")  # 层的名称
        self.norm = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,  # 动量
            epsilon=1e-3,  # 防止除0错误
            name="batch_normalization")  # 层的名称
        self.act = layers.Activation("swish")  # 激活函数

        self.avg = layers.GlobalAveragePooling2D()  # 平均池化
        self.fc = layers.Dense(num_classes,  # 全连接层，输出类别数个数的节点
                               kernel_initializer=DENSE_KERNEL_INITIALIZER)  # 权重初始化器

        if drop_rate > 0:
            self.dropout = layers.Dropout(drop_rate)  # 随机失活操作

    def call(self, inputs, training=None):
        x = self.conv_head(inputs)  # 卷积操作
        x = self.norm(x)  # BN操作
        x = self.act(x)  # 激活函数
        x = self.avg(x)  # 平均池化操作

        if self.dropout:
            x = self.dropout(x, training=training)  # 随机失活操作

        x = self.fc(x)  # 全连接层操作
        return x  # 返回预测的结果特征向量


# 模型的整体实现，根据给定的配置参数构建模型结构。
# 使用官方的预训练权重，参数设置与源码相同
class EfficientNetV2(Model):
    def __init__(self,
                 model_cnf: list,  # 模型配置
                 num_classes: int = 1000,  # 类别数
                 num_features: int = 1280,  # 特征向量长度
                 dropout_rate: float = 0.2,  # 随机失活率
                 drop_connect_rate: float = 0.2,  # 随机失活连接率
                 name: str = None):  # 模型的名称
        super(EfficientNetV2, self).__init__(name=name)

        for cnf in model_cnf:
            assert len(cnf) == 8

        stem_filter_num = model_cnf[0][4]  # 获取干节点卷积核数量
        self.stem = Stem(stem_filter_num)  # 构建干节点层

        total_blocks = sum([i[0] for i in model_cnf])  # 总共的块数
        block_id = 0  # 块编号
        self.blocks = []  # 所有块的集合
        # Builds blocks.
        for cnf in model_cnf:
            repeats = cnf[0]  # 该类型块的重复次数
            op = FusedMBConv if cnf[-2] == 0 else MBConv  # 卷积块类型
            for i in range(repeats):
                # 创建一个块并加入到块集合中
                self.blocks.append(op(kernel_size=cnf[1],
                                      input_c=cnf[4] if i == 0 else cnf[5],  # 输入通道数
                                      out_c=cnf[5],  # 输出通道数
                                      expand_ratio=cnf[3],  # 扩张率
                                      stride=cnf[2] if i == 0 else 1,  # 步长
                                      se_ratio=cnf[-1],  # SE模块中Squeeze的比例
                                      drop_rate=drop_connect_rate * block_id / total_blocks,  # 随机失活连接率
                                      name="blocks_{}".format(block_id)))  # 块的名称
                block_id += 1  # 块编号增加

        self.head = Head(num_features, num_classes, dropout_rate)  # 构建头部层

    # def summary(self, input_shape=(224, 224, 3), **kwargs):
    #     x = Input(shape=input_shape)
    #     model = Model(inputs=[x], outputs=self.call(x, training=True))
    #     return model.summary()

    def call(self, inputs, training=None):
        x = self.stem(inputs, training)  # 干节点操作

        # 调用所有块进行计算
        for _, block in enumerate(self.blocks):
            x = block(x, training=training)

        x = self.head(x, training=training)  # 头部层操作

        return x  # 返回预测结果特征向量


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2,
                           name="efficientnetv2-s")
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3,
                           name="efficientnetv2-m")
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4,
                           name="efficientnetv2-l")
    return model

# m = efficientnetv2_s()
# m.summary()
