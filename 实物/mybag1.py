import os
import shutil

# 原始图像文件夹路径
original_image_folder = 'D:\我的文档\桌面\实物\data'

# 新的图像文件夹路径（用于保存重命名后的图像）
new_image_folder = 'D:\我的文档\桌面\实物\data1'

# 创建新的图像文件夹（如果不存在）
if not os.path.exists(new_image_folder):
    os.makedirs(new_image_folder)

# 获取原始图像文件列表
original_image_files = sorted(os.listdir(original_image_folder))

# 遍历图像文件并重新命名并保存到新的文件夹
for idx, original_image_file in enumerate(original_image_files):
    # 解析原始文件名
    timestamp, angle_part = original_image_file.split('_')
    angle = '{:.4f}'.format(float(angle_part.split('.')[0] + '.' + angle_part.split('.')[1]))  # 提取并保留四位小数的角速度值
    
    # 构建新文件名
    new_filename = f"{idx}_{angle}.jpg"
    
    # 构建新文件名
    new_filename = f"{idx}_{angle}.jpg"
    
    # 构建原始文件路径和新文件路径
    original_filepath = os.path.join(original_image_folder, original_image_file)
    new_filepath = os.path.join(new_image_folder, new_filename)
    
    # 复制并重命名文件到新的文件夹
    shutil.copyfile(original_filepath, new_filepath)
    
    print(f"Copied and renamed {original_image_file} to {new_filename} in {new_image_folder}")
