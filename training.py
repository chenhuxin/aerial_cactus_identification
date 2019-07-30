import tensorflow as tf
import pandas as pd
import model
import data_process
import os


w = 32
h = 32
batch_size = 32
capacity = 256
max_steps = 2000
train_dir = '../aerial_cactus_identification/data/train/'
model_path = '../aerial_cactus_identification/model'
model_name = 'cactus.ckpt'
train_data = pd.read_csv('../aerial_cactus_identification/data/train.csv')

input_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, w, h, 3], name='x_input')
output = tf.placeholder(dtype=tf.int64, shape=[batch_size], name='y_output')
is_training = tf.placeholder(dtype=tf.bool)
lr = tf.placeholder(dtype=tf.float32)
logit = model.resnet50(input_image, is_training)
sum_loss = model.total_loss(prediction=logit, labels=output)
tf.summary.scalar('loss', sum_loss)
acc = model.accuracy(pred=logit, labels=output)
tf.summary.scalar('accuracy', acc)
global_step = tf.Variable(0, trainable=False)
train_op = model.train(lr, sum_loss, global_step)
coord = tf.train.Coordinator()
merged = tf.summary.merge_all()
saver = tf.train.Saver()


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('../aerial_cactus_identification/logs/', graph=sess.graph)
    image_list = []
    label_list = []
    images = train_data['id'].values
    for image_id in images:
        image_list.append(train_dir + image_id)
        label_list.append(train_data[train_data['id'] == image_id]['has_cactus'].values[0])
    image_train, label_train = data_process.get_batch(image_list[:15001], label_list[:15001], w, h, batch_size, capacity)
    image_valid, label_valid = data_process.get_batch(image_list[15000:], label_list[15000:], w, h, batch_size, capacity)
    learning_rate = model.learning_rate_schedule(max_steps)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(max_steps):
        train_image, train_label = sess.run([image_train, label_train])
        _, loss, steps = sess.run([train_op, sum_loss, global_step],
                                  feed_dict={input_image: train_image, output: train_label,
                                             lr: learning_rate, is_training: True})
        prec = sess.run(acc, feed_dict={input_image: train_image, output: train_label, is_training: False})
        if steps % 10 == 0:
            print("after {:d} training step ,loss is {:0.3f}, accuracy is {:0.3f}".format(steps, loss, prec))

        if steps % 1000 == 0:
            result = sess.run(merged, feed_dict={input_image: train_image, output: train_label,
                                                 lr: learning_rate, is_training: False})
            saver.save(sess, os.path.join(model_path, model_name), global_step=steps)
            writer.add_summary(result, steps)
            writer.close()
    valid_image, valid_label = sess.run([image_valid, label_valid])
    valid_acc = sess.run(acc, feed_dict={input_image: valid_image, output: valid_label, is_training: False})
    print('after {:d} training step, the valid accurancy is {:0.3f}'.format(max_steps, valid_acc))
    coord.request_stop()
    coord.join(threads)
