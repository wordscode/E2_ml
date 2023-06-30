import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, send_file

from model import efficientnetv2_s as create_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 将CORS中间件添加到Flask应用程序中

num_classes = 17  # 分类总数
img_size = {"s": 384, "m": 480, "l": 480}
num_model = "s"  # 使用哪个模型
im_height = im_width = img_size[num_model]  # 图片的宽和高

# 读取类别字典
json_path = './class_indices.json'
assert os.path.exists(json_path), "文件: '{}' 不存在。".format(json_path)  # 判断文件是否存在
with open(json_path, "r") as f:
    class_indict = json.load(f)

# 创建模型
model = create_model(num_classes=num_classes)

weights_path = './save_weights/efficientnetv2.ckpt'
assert len(glob.glob(weights_path + "*")), "找不到 {}".format(weights_path)  # 判断文件是否存在
model.load_weights(weights_path)  # 加载预训练模型权重


def preprocess_image(image):
    image = image.resize((im_width, im_height))
    image = np.array(image).astype(np.float32)
    if len(image.shape) == 2:  # 处理灰度图像
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # 处理带有Alpha通道的图像
        image = image[:, :, :3]
    image = (image / 255. - 0.5) / 0.5
    image = np.expand_dims(image, 0)
    return image

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '请求中没有找到图像'})

    image_file = request.files['image']
    image = Image.open(image_file)

    image_data = preprocess_image(image)

    result = np.squeeze(model.predict(image_data))
    result = tf.keras.layers.Softmax()(result)
    predict_class = np.argmax(result)
    print(predict_class)

    prediction = {
        'class': class_indict[str(predict_class)],
        'probability': float(result[predict_class])
    }
    return jsonify(prediction)


if __name__ == '__main__':
    app.run()
