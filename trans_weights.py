from model import *

# 从EfficientNetV2模型的检查点文件中加载权重并保存为HDF5格式的权重文件
def main(ckpt_path: str,
         model_name: str,
         model: tf.keras.Model):
    # 创建一个字典，用于存储模型中的权重变量
    var_dict = {v.name.split(':')[0]: v for v in model.weights}

    # 加载检查点文件
    reader = tf.train.load_checkpoint(ckpt_path)
    # 获取检查点文件中的变量名和形状的映射关系
    var_shape_map = reader.get_variable_to_shape_map()

    # 遍历权重变量字典中的每个键值对
    for key, var in var_dict.items():
        # 构建在检查点文件中对应的键
        key_ = model_name + "/" + key
        # 替换键中的"batch_normalization"为"tpu_batch_normalization"
        key_ = key_.replace("batch_normalization", "tpu_batch_normalization")

        # 检查键是否存在于检查点文件中
        if key_ in var_shape_map:
            # 检查权重的形状是否与检查点文件中的形状匹配
            if var_shape_map[key_] != var.shape:
                # 形状不匹配，输出错误信息
                msg = "shape mismatch: {}".format(key)
                print(msg)
            else:
                # 形状匹配，从检查点文件中获取权重值，并赋值给模型的权重变量
                var.assign(reader.get_tensor(key_), read_value=False)
        else:
            # 在检查点文件中找不到对应的权重值，输出错误信息
            msg = "Not found {} in {}".format(key, ckpt_path)
            print(msg)

    # 将加载的权重保存为HDF5格式的权重文件
    model.save_weights("./{}.h5".format(model_name))


if __name__ == '__main__':
    # 创建EfficientNetV2-S模型对象
    model = efficientnetv2_s()
    model.build((1, 224, 224, 3))

    # 调用main函数加载权重并保存
    main(ckpt_path="./efficientnetv2-s-21k-ft1k/",
         model_name="efficientnetv2-s",
         model=model)

    # 可选：加载其他模型的权重并保存
    # model = efficientnetv2_m()
    # model.build((1, 224, 224, 3))
    # main(ckpt_path="./efficientnetv2-m-21k-ft1k/model",
    #      model_name="efficientnetv2-m",
    #      model=model)

    # model = efficientnetv2_l()
    # model.build((1, 224, 224, 3))
    # main(ckpt_path="./efficientnetv2-l-21k-ft1k/model",
    #      model_name="efficientnetv2-l",
    #      model=model)
