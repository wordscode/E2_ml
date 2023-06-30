import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    num_classes = 17  # 分类总数

    img_size = {"s": 384,  # 不同模型对应的图片尺寸
                "m": 480,
                "l": 480}
    num_model = "s"  # 使用哪个模型
    im_height = im_width = img_size[num_model]  # 图片的宽和高

    # 加载图像
    img_path = "./test.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)  # 判断文件是否存在
    img = Image.open(img_path)
    # 调整图像大小
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # 读取图像,img转换为NumPy数组类型,转换为32位浮点型
    img = np.array(img).astype(np.float32)

    # 预处理图像,归一化处理
    img = (img / 255. - 0.5) / 0.5

    # 将图片加入到一个批次中
    img = (np.expand_dims(img, 0))

    # 读取类别字典
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)  # 判断文件是否存在

    # 从JSON 文件中读取数据，并将其转换为 Python 对象
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(num_classes=num_classes)

    weights_path = './save_weights/efficientnetv2.ckpt'
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)  # 判断文件是否存在
    model.load_weights(weights_path)  # 加载预训练模型权重

    result = np.squeeze(model.predict(img))  # 对图像进行推理
    result = tf.keras.layers.Softmax()(result)  # 每个类别的概率分布
    predict_class = np.argmax(result)  # 找到概率最大的类别

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])  # 输出预测结果
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i]))  # 输出每个类别的概率
    plt.show()  # 显示图片和预测结果


if __name__ == '__main__':
    main()
