import os
import shutil
# 将数据集划分为17类

# 原始图片路径
source_path = '../17flowers'

# 目标文件夹路径
target_path = '../17'

# 子目录名称列表
directory_names = ['Daffodil', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
                   'Iris', 'Tigerlily', 'Tulip', 'Fritillary', 'Sunflower',
                   'Daisy', "Colts'Foot", 'Dandelion', 'Cowslip', 'Buttercup',
                   'Windflower', 'Pansy']


# 创建子目录
for directory_name in directory_names:
    directory_path = os.path.join(target_path, directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# 将每80张图片放入相应的子目录中
images_per_directory = 80
for i in range(len(directory_names)):
    directory_name = directory_names[i]
    start_index = i * images_per_directory

    # 获取要移动的文件名列表
    file_names = sorted(
        [f for f in os.listdir(source_path) if not f.startswith('.') and not f.startswith('files.txt')])[
                 start_index:start_index + images_per_directory]

    # 将文件移动到对应的子目录中
    for file_name in file_names:
        source_file_path = os.path.join(source_path, file_name)
        destination_file_path = os.path.join(target_path, directory_name, file_name)
        shutil.copy(source_file_path, destination_file_path)
