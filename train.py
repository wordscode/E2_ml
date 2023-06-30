import os
import sys
import datetime
import io

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import efficientnetv2_s as create_model
from utils import generate_ds
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import json
import pandas as pd
from sklearn.metrics import classification_report



# cd C:\Users\czk\PycharmProjects\pythonProject2\Test11_efficientnetV2
# tensorboard --logdir=./logs/


def main():
    data_root = "../17/"  # 获取数据根目录

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    img_size = {"s": [300, 384],  # 训练尺寸，测试尺寸
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    batch_size = 8
    epochs = 1
    num_classes = 17
    freeze_layers = True
    initial_lr = 0.01

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    test_writer = tf.summary.create_file_writer(os.path.join(log_dir, "test"))

    # 带有数据增强的数据生成器
    train_ds, test_ds = generate_ds(data_root,
                                    train_im_height=img_size[num_model][0],
                                    train_im_width=img_size[num_model][0],
                                    test_im_height=img_size[num_model][1],
                                    test_im_width=img_size[num_model][1],
                                    batch_size=batch_size)

    # 创建模型
    model = create_model(num_classes=num_classes)
    model.build((1, img_size[num_model][0], img_size[num_model][0], 3))

    # 加载权重
    pre_weights_path = './tf_efficientnetv2/efficientnetv2-s.h5'
    assert os.path.exists(pre_weights_path), "无法找到{}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # 冻结底层网络层
    if freeze_layers:
        unfreeze_layers = "head"
        for layer in model.layers:
            if unfreeze_layers not in layer.name:
                layer.trainable = False
            else:
                print("训练 {}".format(layer.name))

    model.summary()

    # 使用低级API进行训练
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)  # 稀疏分类交叉熵损失函数，True 表示模型的输出是未经过 softmax 处理的原始预测结果。
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr,
                                        momentum=0.9)  # 随机梯度下降优化器SGD，momentum动量系数为 0.9，动量优化算法加速模型的训练过程并提高收敛性能

    train_loss = tf.keras.metrics.Mean(name='train_loss')  # 损失值
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  # 准确率

    # 测试
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    test_precision = tf.keras.metrics.Precision(name='test_precision')
    test_recall = tf.keras.metrics.Recall(name='test_recall')
    misclassified_images = []
    misclassified_labels = []
    test_labels_list = []
    predicted_labels_list = []

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)  # 梯度
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 将梯度应用到模型的可训练变量上，从而更新模型的参数。
        # 更新训练损失和准确率的指标对象
        train_loss(loss)
        train_accuracy(train_labels, output)

    best_train_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # 清除历史信息
        train_accuracy.reset_states()  # 清除历史信息

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            #  打印训练进度
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # 写入训练损失和准确率
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

        # 只保存基于训练准确率的最佳权重
        if train_accuracy.result() > best_train_acc:
            best_train_acc = train_accuracy.result()
            save_name = "./save_weights/efficientnetv2.ckpt"
            model.save_weights(save_name, save_format="tf")

    # 测试
    tf.summary.experimental.set_step(0)

    @tf.function
    def test_step(test_images, test_labels):
        output = model(test_images, training=False)
        loss = loss_object(test_labels, output)

        test_loss(loss)
        test_accuracy(test_labels, output)

        # 将测试标签转换为int64
        test_labels = tf.cast(test_labels, tf.int64)
        # 计算精确度和召回率
        predicted_labels = tf.argmax(output, axis=1)
        test_precision.update_state(test_labels, predicted_labels)
        test_recall.update_state(test_labels, predicted_labels)

        # 存储错误分类的图像
        misclassified_indices = tf.where(predicted_labels != test_labels)
        misclassified_images = tf.gather(test_images, misclassified_indices)
        misclassified_labels = tf.gather(test_labels, misclassified_indices)

        return predicted_labels,misclassified_images, misclassified_labels




    test_bar = tqdm(test_ds, file=sys.stdout)
    for images, labels in test_bar:
        predicted_labels,mis_images, mis_labels = test_step(images, labels)
        test_labels_list.extend(labels.numpy())
        predicted_labels_list.extend(predicted_labels.numpy())
        misclassified_images.append(mis_images)
        misclassified_labels.append(mis_labels)

        # 打印测试进度
        test_bar.desc = "test epoch loss:{:.3f}, acc:{:.3f}".format(test_loss.result(), test_accuracy.result())

    test_labels = tf.constant(test_labels_list)
    predicted_labels = tf.constant(predicted_labels_list)

    test_precision_value = precision_score(test_labels, predicted_labels, average='macro')
    test_recall_value = recall_score(test_labels, predicted_labels, average='macro')
    classification_rep = classification_report(test_labels, predicted_labels)

    # 将测试指标写入test_writer
    with test_writer.as_default():
        tf.summary.scalar("loss", test_loss.result(), step=0)
        tf.summary.scalar("accuracy", test_accuracy.result(), step=0)
        tf.summary.scalar("precision", test_precision_value, step=0)
        tf.summary.scalar("recall", test_recall_value, step=0)
        print(classification_rep)
        tf.summary.text("classification_report", tf.convert_to_tensor(classification_rep), step=0)

        # 计算混淆矩阵
        confusion_mat = confusion_matrix(test_labels, predicted_labels)
        # 类别名称的列表，用于显示在混淆矩阵的标签旁边。
        class_names = [str(i) for i in range(num_classes)]

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                    ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = image.reshape((1,) + image.shape)

        plt.close()
        tf.summary.image("confusion_matrix", image, step=0)


    # 连接错误分类的图像和标签
    misclassified_images = tf.concat(misclassified_images, axis=0)
    misclassified_labels = tf.concat(misclassified_labels, axis=0)

    # 创建保存错误分类图像的目录
    misclassified_dir = "./misclassified_images/"
    os.makedirs(misclassified_dir, exist_ok=True)

    # 保存错误分类的图像
    for i in range(len(misclassified_images)):
        image = tf.squeeze(misclassified_images[i], axis=0)  # 压缩维度
        img = tf.keras.preprocessing.image.array_to_img(image)
        # 获取实际类别和错误分类
        actual_label = misclassified_labels[i].numpy()
        misclassified_label = tf.argmax(model(tf.expand_dims(image, axis=0)), axis=1).numpy()[0]
        with open('class_indices.json', 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        actual_name=json_data[str(actual_label[0])]
        misclassified_name=json_data[str(misclassified_label)]
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")# 获取当前时间
        file_name = f"{current_time}_actual_{actual_name}_misclassified_{misclassified_name}.png"
        img.save(os.path.join(misclassified_dir, file_name))


if __name__ == '__main__':
    main()
