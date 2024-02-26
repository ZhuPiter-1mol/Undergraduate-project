import rosbag
import cv2
import os
from cv_bridge import CvBridge
import numpy as np

# 读取ROS bag文件
bag = rosbag.Bag('/path/to/your/rosbag/file.bag')

# 订阅RGB图像话题和角速度话题
image_topic = '/camera/image_raw'
velocity_topic = '/velocity'

# 存储RGB图像和角速度数据的字典
image_data = {}
velocity_data = {}

# 处理ROS bag中的消息，先处理角速度话题
for topic, msg, t in bag.read_messages(topics=[velocity_topic]):
    if topic == velocity_topic:
        # 处理角速度消息
        velocity_timestamp = msg.header.stamp.to_sec()
        velocity_value = msg.data
        velocity_data[velocity_timestamp] = velocity_value

# 用于角速度插值
def interpolate_velocity(timestamp):
    closest_timestamp = min(velocity_data.keys(), key=lambda x: abs(x - timestamp))
    return velocity_data[closest_timestamp]

# 创建存储图像的文件夹
image_folder = '/path/to/save/images'
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 处理ROS bag中的消息，处理图像话题
for topic, msg, t in bag.read_messages(topics=[image_topic]):
    if topic == image_topic:
        # 处理RGB图像消息
        cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_timestamp = msg.header.stamp.to_sec()
        image_filename = f"{image_timestamp}_{interpolate_velocity(image_timestamp):.4f}.jpg"
        cv2.imwrite(os.path.join(image_folder, image_filename), cv_image)

# 关闭ROS bag文件
bag.close()
