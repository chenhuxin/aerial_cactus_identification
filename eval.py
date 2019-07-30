import tensorflow as tf
import model
import os
import pandas as pd


test_dir = '../aerial_cactus_identification/data/test/'
model_path = '../aerial_cactus_identification/model'

test_file = os.listdir(test_dir)
image_list = []
ids = []
for image_id in test_file:
    image_contents = tf.read_file(test_dir+image_id)  # 读取图片
    image_decode = tf.image.decode_jpeg(image_contents, channels=3)  # 图片解码
    image = tf.image.per_image_standardization(image_decode)  # 标准化图片
    image_list.append(image)
    ids.append(image_id)
image=tf.cast(image_list,dtype=tf.float32)


input_image = tf.placeholder(dtype=tf.float32, shape=[4000, 32, 32, 3])
is_training = tf.placeholder(dtype=tf.bool)
logit = model.resnet50(input_image, is_training)
pred = tf.argmax(logit, 1)
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        inp=sess.run(image)
        prediction = sess.run(pred, feed_dict={input_image: inp, is_training: False})
        sample_submission = pd.DataFrame({'id': ids, 'has_cactus': prediction})
        sample_submission.to_csv('cactus.csv', index=False, columns=['id', 'has_cactus'])
