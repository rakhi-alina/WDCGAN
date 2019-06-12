import numpy as np
import tensorflow as tf


def sample_noise(size, mu, sigma):
    return np.random.normal(mu, sigma, size=size)
    #return np.random.uniform(-1., 1., size=size)


def generator(z, trainable, reuse=False):
    initializer = tf.random_normal_initializer(0, 0.02)
    
    with tf.variable_scope('generator', reuse=reuse):

        deconv1 = tf.layers.conv2d_transpose(z, 512, 4, 1, padding='VALID', use_bias=False, kernel_initializer=initializer)
        bn1 = tf.layers.batch_normalization(deconv1, training=trainable)
        relu1 = tf.nn.relu(bn1)

        deconv2 = tf.layers.conv2d_transpose(relu1, 256, 4, 2, padding='SAME', use_bias=False, kernel_initializer=initializer)
        bn2 = tf.layers.batch_normalization(deconv2, training=trainable)
        relu2 = tf.nn.relu(bn2)

        deconv3 = tf.layers.conv2d_transpose(relu2, 128, 4, 2, padding='SAME', use_bias=False, kernel_initializer=initializer)
        bn3 = tf.layers.batch_normalization(deconv3, training=trainable)
        relu3 = tf.nn.relu(bn3)

        deconv4 = tf.layers.conv2d_transpose(relu3, 64, 4, 2, padding='SAME', use_bias=False, kernel_initializer=initializer)
        bn4 = tf.layers.batch_normalization(deconv4, training=trainable)
        relu4 = tf.nn.relu(bn4)

        deconv5 = tf.layers.conv2d_transpose(relu4, 3, 4, 2, padding='SAME', use_bias=False, kernel_initializer=initializer)
        output = tf.tanh(deconv5)

        print(z)
        print(relu1)
        print(relu2)
        print(relu3)
        print(relu4)
        print(output)
   
    return output


def discriminator(x, trainable, reuse=False):
    initializer = tf.random_normal_initializer(0, 0.02)
    
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 64, 4, 2, 'SAME', kernel_initializer=initializer)
        lrelu1 = tf.nn.leaky_relu(conv1)

        conv2 = tf.layers.conv2d(lrelu1, 128, 4, 2, 'SAME', kernel_initializer=initializer)
        bn2 = tf.layers.batch_normalization(conv2, training=trainable)
        lrelu2 = tf.nn.leaky_relu(bn2)

        conv3 = tf.layers.conv2d(lrelu2, 256, 4, 2, 'SAME', kernel_initializer=initializer)
        bn3 = tf.layers.batch_normalization(conv3, training=trainable)
        lrelu3 = tf.nn.leaky_relu(bn3)
        
        conv4 = tf.layers.conv2d(lrelu3, 512, 4, 2, 'SAME', kernel_initializer=initializer)
        bn4 = tf.layers.batch_normalization(conv4, training=trainable)
        lrelu4 = tf.nn.leaky_relu(bn4)

        conv5 = tf.layers.conv2d(lrelu4, 1, 4, 1, 'VALID', kernel_initializer=initializer)
        flattened = tf.layers.flatten(conv5)
        probability = tf.sigmoid(flattened)
    
    if not reuse:
        print(x)
        print(lrelu1)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(conv5)
        print(flattened)
        print(probability)

    return probability, flattened