import numpy as np
import tensorflow as tf


def sample_noise(size, mu, sigma):
    return np.random.normal(mu, sigma, size=size)
    #return np.random.uniform(-1., 1., size=size)


def generator(z, trainable, reuse=False):
    #initializer = tf.random_normal_initializer(0, 0.02)
    
    with tf.variable_scope('generator', reuse=reuse):

        fc1 = tf.layers.dense(z, 4*4*1024, use_bias=False)
        bn1 = tf.layers.batch_normalization(fc1, training=trainable)
        relu1 = tf.nn.relu(bn1)

        reshaped = tf.reshape(relu1, [-1, 4, 4, 1024])

        deconv2 = tf.layers.conv2d_transpose(reshaped, 512, 5, 2, padding='SAME', use_bias=False)
        bn2 = tf.layers.batch_normalization(deconv2, training=trainable)
        lrelu2 = tf.nn.leaky_relu(bn2)

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 256, 5, 2, padding='SAME', use_bias=False)
        bn3 = tf.layers.batch_normalization(deconv3, training=trainable)
        lrelu3 = tf.nn.leaky_relu(bn3)

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 128, 5, 2, padding='SAME', use_bias=False)
        bn4 = tf.layers.batch_normalization(deconv4, training=trainable)
        lrelu4 = tf.nn.leaky_relu(bn4)

        deconv5 = tf.layers.conv2d_transpose(lrelu4, 3, 5, 2, padding='SAME', use_bias=False)
        output = tf.tanh(deconv5)

        print(z)
        print(relu1)
        print(reshaped)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(output)
   
    return output


def discriminator(x, trainable, reuse=False):

    
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 64, 5, 2, 'SAME')
        lrelu1 = tf.nn.leaky_relu(conv1)
        drop1 = tf.layers.dropout(lrelu1, rate=0.3)

        conv2 = tf.layers.conv2d(drop1, 128, 5, 2, 'SAME')
        lrelu2 = tf.nn.leaky_relu(conv2)
        drop2 = tf.layers.dropout(lrelu2, rate=0.3)

        conv3 = tf.layers.conv2d(drop2, 256, 5, 2, 'SAME')
        lrelu3 = tf.nn.leaky_relu(conv3)
        drop3 = tf.layers.dropout(lrelu3, rate=0.3)

        flattened = tf.layers.flatten(drop3)
        output = tf.layers.dense(flattened, 1)
    
    if not reuse:
        print(x)
        print(drop1)
        print(drop2)
        print(drop3)
        print(flattened)
        print(output)

    return output