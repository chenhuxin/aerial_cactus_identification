import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf

'''
train_dir = '../aerial_cactus_identification/data/train/'
test_dir = '../aerial_cactus_identification/data/test/'

train_data = pd.read_csv('../aerial_cactus_identification/data/train.csv')

image_list = []
label_list = []
images = train_data['id'].values
for image_id in images:
    image_list.append(train_dir + image_id)
    label_list.append(train_data[train_data['id'] == image_id]['has_cactus'].values[0])
print(image_list, label_list)
'''

# 输入数据处理 1、输入文件列表 2、输入文件队列 3、读数据和数据预处理 4、整理成batch

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])  # 输入文件队列，有一个参数shuffle=True时后面就是用tf.train.batch否则用tf.train.shuffle_batch
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # 读取图片
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 图片解码
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  # 预处理图片
    image = tf.image.per_image_standardization(image)  # 标准化图片

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

'''
BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 32
IMG_H = 32
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
sess.close()
'''