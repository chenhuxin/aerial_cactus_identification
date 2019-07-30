import tensorflow as tf
import tensorflow.contrib.slim as slim


def resnet_arg_scope(is_training=True, batch_norm_decay=0.997, batch_norm_epsilon=1e-5,
                     batch_norm_scale=True, weight_decay=1e-4):
    batch_norm_params = {'is_training': is_training,
                         'decay': batch_norm_decay,
                         'epsilon': batch_norm_epsilon,
                         'scale': batch_norm_scale,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}

    with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.glorot_uniform_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def identity_block(x_input, channels, f, stage, block, is_training):
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        x_shortcut = x_input
        c1, c2, c3 = channels
        x = slim.conv2d(x_input, c1, [1, 1], stride=1, padding='Valid', scope=conv_name_base + '2a')
        x = slim.conv2d(x, c2, [f, f], stride=1, padding='SAME', scope=conv_name_base + '2b')
        x = slim.conv2d(x, c3, [1, 1], stride=1, padding='Valid', activation_fn=None, scope=conv_name_base + '2c')
        out_tensor = tf.add(x, x_shortcut)
        x = tf.nn.relu(out_tensor)
    return x


def convolution_block(x_input, channels, f, s, stage, block, is_training):
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        conv_name_base = 'res' + str(stage) + block + '_branch'
        x_shortcut = x_input
        c1, c2, c3 = channels
        x = slim.conv2d(x_input, c1, [1, 1], stride=s, padding='Valid', scope=conv_name_base + '2a')
        x = slim.conv2d(x, c2, [f, f], stride=1, padding='SAME', scope=conv_name_base + '2b')
        x = slim.conv2d(x, c3, [1, 1], stride=1, padding='Valid', activation_fn=None, scope=conv_name_base + '2c')
        out_tensor = slim.conv2d(x_shortcut, c3, [1, 1], stride=s, padding='Valid',
                                 activation_fn=None, scope=conv_name_base + '1')
        x = tf.nn.relu(tf.add(out_tensor, x))
        return x


def resnet50(input_image, is_training):
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        x = slim.conv2d(input_image, 64, [7, 7], stride=2, padding='SAME', scope='conv1')
        x = slim.max_pool2d(x, [3, 3], stride=2, scope='pool1')

    x = convolution_block(x, [64, 64, 256], f=3, s=1, stage=2, block='a', is_training=is_training)
    x = identity_block(x, [64, 64, 256], f=3, stage=2, block='b', is_training=is_training)
    x = identity_block(x, [64, 64, 256], f=3, stage=2, block='c', is_training=is_training)

    x = convolution_block(x, [128, 128, 512], f=3, s=2, stage=3, block='a', is_training=is_training)
    x = identity_block(x, [128, 128, 512], f=3, stage=3, block='b', is_training=is_training)
    x = identity_block(x, [128, 128, 512], f=3, stage=3, block='c', is_training=is_training)
    x = identity_block(x, [128, 128, 512], f=3, stage=3, block='d', is_training=is_training)

    x = convolution_block(x, [256, 256, 1024], f=3, s=2, stage=4, block='a', is_training=is_training)
    x = identity_block(x, [256, 256, 1024], f=3, stage=4, block='b', is_training=is_training)
    x = identity_block(x, [256, 256, 1024], f=3, stage=4, block='c', is_training=is_training)
    x = identity_block(x, [256, 256, 1024], f=3, stage=4, block='d', is_training=is_training)
    x = identity_block(x, [256, 256, 1024], f=3, stage=4, block='e', is_training=is_training)
    x = identity_block(x, [256, 256, 1024], f=3, stage=4, block='f', is_training=is_training)

    x = convolution_block(x, [512, 512, 2048], f=3, s=2, stage=5, block='a', is_training=is_training)
    x = identity_block(x, [512, 512, 2048], f=3, stage=5, block='b', is_training=is_training)
    x = identity_block(x, [512, 512, 2048], f=3, stage=5, block='c', is_training=is_training)
    global_average_shape = x.get_shape().as_list()
    x = tf.nn.avg_pool(x, ksize=[1, global_average_shape[1], global_average_shape[2], 1],
                       strides=[1, 1, 1, 1], padding='VALID')
    x = slim.flatten(x, scope='global_avg')
    logit = slim.fully_connected(x, 2, activation_fn=None, weights_initializer=tf.glorot_uniform_initializer(),
                                 biases_initializer=tf.constant_initializer(0.1),
                                 weights_regularizer=slim.l2_regularizer(1e-4), scope='fc')

    return logit


def total_loss(prediction, labels):
    with tf.variable_scope('losses'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
        cross_entropy = tf.reduce_mean(cross_entropy)
        regularizer_loss = tf.add_n(tf.losses.get_regularization_losses())
        sum_loss = cross_entropy + regularizer_loss
        return sum_loss


def accuracy(pred, labels):
    with tf.variable_scope('precision'):
        correct = tf.equal(tf.argmax(pred, 1), labels)
        correct = tf.cast(correct, tf.float32)
        acc = tf.reduce_mean(correct)
        return acc


def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    else:
        return 0.001


def train(lr, losses, global_step):
    with tf.variable_scope('training'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr).minimize(losses, global_step=global_step)
        return train_op
