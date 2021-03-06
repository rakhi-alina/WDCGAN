import numpy as np
import tensorflow as tf


'''def sample_noise(size, mu, sigma):
    return np.random.normal(mu, sigma, size=size)
    #return np.random.uniform(-1., 1., size=size)'''

def sample_noise(batch_size, num_dims):
    return tf.random_normal([batch_size, num_dims])


def generator(z, trainable, reuse=False):
    
    with tf.variable_scope('generator', reuse=reuse):
        size1, size2, size3 = 4, 4, 1024
        fc1 = tf.reshape(tf.layers.dense(z, size1*size2*size3, use_bias=False), [-1, size1, size2, size3])
        lrelu0 = tf.nn.leaky_relu(tf.layers.batch_normalization(fc1, training=trainable))

        deconv1 = tf.layers.conv2d_transpose(lrelu0, 512, 5, 2, padding='SAME', use_bias=False)
        lrelu1 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv1, training=trainable))

        deconv2 = tf.layers.conv2d_transpose(lrelu1, 256, 5, 2, padding='SAME', use_bias=False)
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv2, training=trainable))

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, 5, 2, padding='SAME', use_bias=False)
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv3, training=trainable))

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 64, 5, 2, padding='SAME', use_bias=False)
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv4, training=trainable))

        output = tf.tanh(tf.layers.conv2d_transpose(lrelu4, 3, 5, 2, padding='SAME', use_bias=False))

        print(z)
        print(lrelu0)
        print(lrelu1)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(output)
        print()
   
    return output


def discriminator(x, reuse=False):
    
    with tf.variable_scope('discriminator', reuse=reuse):
        lrelu1 = tf.nn.leaky_relu(tf.layers.conv2d(x, 64, 5, 2, 'SAME'))
        lrelu2 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME'))
        lrelu3 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu2, 256, 5, 2, 'SAME'))
        lrelu4 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu3, 512, 5, 2, 'SAME'))
        lrelu5 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu4, 1024, 5, 2, 'SAME'))
        output = tf.layers.dense(tf.layers.flatten(lrelu5), 1)
    
    if not reuse:
        print(x)
        print(lrelu1)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(lrelu5)
        print(output)

    return output