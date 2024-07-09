import os
import random
import shutil

from PIL.Image import LANCZOS
from PIL import Image
from tqdm import tqdm


def get_txt_file():
    train_folder = 'data\\Image_restoration\\LL_dataset\\Construction\\train\\low_M'
    train_file_path = "data\\Image_restoration\\LL_dataset\\Construction\\train\\Construction_train_M.txt"
    val_file_path = "data\\Image_restoration\\LL_dataset\\Construction\\val\\Construction_val_M.txt"

    filenames = list()
    for filename in os.listdir(train_folder):
        filenames.append(filename)

    # 随机打乱列表
    random.shuffle(filenames)

    # 计算80%的位置
    split_index = int(len(filenames) * 0.85)

    # 切分列表
    list_80_percent = filenames[:split_index]
    list_20_percent = filenames[split_index:]

    with open(train_file_path, 'w') as file:
        for img_name in list_80_percent:
            file.write(os.path.join("\\low_M", img_name) + '\n')

    with open(val_file_path, 'w') as file:
        for img_name in list_20_percent:
            file.write(os.path.join("\\low_M", img_name) + '\n')

def resize_img():
    source_folder = 'data\\data\\low'
    target_folder = 'data\\data\\low_'

    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for file_name in tqdm(os.listdir(source_folder)):
        if file_name.endswith('.png') or file_name.endswith('.jpeg'):  # 检查文件是否为JPEG
            # 构造完整的文件路径
            file_path = os.path.join(source_folder, file_name)

            # 打开并加载图像
            with Image.open(file_path) as img:
                # 调整图像大小
                img_resized = img.resize((600, 400), LANCZOS)

                # 构造新的文件路径
                new_file_path = os.path.join(target_folder, file_name)

                # 保存调整大小后的图像
                img_resized.save(new_file_path)

    print("所有图片处理完成。")


def copy_files():
    src_dir = "data\\data\\high_"
    src_dir_ = "data\\data\\low_"
    dst_dir = "data\\Image_restoration\\LL_dataset\\Construction\\train\\high_M"
    dst_dir_ = "data\\Image_restoration\\LL_dataset\\Construction\\train\\low_M"
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if not os.path.exists(dst_dir_):
        os.makedirs(dst_dir_)

    filenames = list()
    # 遍历源目录
    for filename in os.listdir(src_dir):
        filenames.append(filename)

    # 随机打乱列表
    random.shuffle(filenames)

    # 计算80%的位置
    split_index = int(len(filenames) * 0.4)

    # 切分列表
    random_files = filenames[:split_index]

    for file in random_files:
        src_file = os.path.join(src_dir, file)
        src_file_ = os.path.join(src_dir_, file)
        # 确保是文件而非目录
        if os.path.isfile(src_file):
            dst_file = os.path.join(dst_dir, file)

            # 复制文件
            shutil.copy(src_file, dst_file)
            print(f"Copied '{src_file}' to '{dst_file}'")

        if os.path.isfile(src_file_):
            dst_file_ = os.path.join(dst_dir_, file)

            # 复制文件
            shutil.copy(src_file_, dst_file_)
            print(f"Copied '{src_file_}' to '{dst_file_}'")


if __name__ == '__main__':
    resize_img()
