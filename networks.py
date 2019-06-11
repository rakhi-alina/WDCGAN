import numpy as np
import tensorflow as tf


def sample_noise(batch_size, size, mu, sigma):
    return np.random.normal(mu, sigma, size=[batch_size, size])
    #return np.random.uniform(-1., 1., size=[batch_size, size])


def generator(z, trainable, reuse=False):
    
    with tf.variable_scope('generator', reuse=reuse):
        
        fc1 = tf.layers.dense(z, 4*4*1024)
        
        reshaped = tf.reshape(fc1, (-1, 4, 4, 1024))
        bn0 = tf.layers.batch_normalization(reshaped, training=trainable)
        lrelu0 = tf.nn.leaky_relu(bn0)
        
        deconv1 = tf.layers.conv2d_transpose(lrelu0, 512, 5, 1, padding='VALID')
        bn1 = tf.layers.batch_normalization(deconv1, training=trainable)
        lrelu1 = tf.nn.leaky_relu(bn1)
        
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 256, 5, 2, padding='SAME')
        bn2 = tf.layers.batch_normalization(deconv2, training=trainable)
        lrelu2 = tf.nn.leaky_relu(bn2)
        
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, 5, 2, padding='SAME')
        bn3 = tf.layers.batch_normalization(deconv3, training=trainable)
        lrelu3 = tf.nn.leaky_relu(bn3)

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 64, 5, 2, padding='SAME')
        bn4 = tf.layers.batch_normalization(deconv4, training=trainable)
        lrelu4 = tf.nn.leaky_relu(bn4)
        
        deconv5 = tf.layers.conv2d_transpose(lrelu4, 3, 5, 2, padding='SAME')
        output = tf.tanh(deconv5)
        
        print(z)
        print(fc1)
        print(lrelu0)
        print(lrelu1)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(output)
        
    return output
    

def discriminator(x, trainable, reuse=False):
    
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 64, 5, 2, 'SAME')
        lrelu1 = tf.nn.leaky_relu(conv1)

        conv2 = tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME')
        bn2 = tf.layers.batch_normalization(conv2, training=trainable)
        lrelu2 = tf.nn.leaky_relu(bn2)

        conv3 = tf.layers.conv2d(lrelu2, 256, 5, 2, 'SAME')
        bn3 = tf.layers.batch_normalization(conv3, training=trainable)
        lrelu3 = tf.nn.leaky_relu(bn3)
        
        conv4 = tf.layers.conv2d(lrelu3, 512, 5, 2, 'SAME')
        bn4 = tf.layers.batch_normalization(conv4, training=trainable)
        lrelu4 = tf.nn.leaky_relu(bn4)

        conv5 = tf.layers.conv2d(lrelu4, 512, 5, 2, 'SAME')
        bn5 = tf.layers.batch_normalization(conv5, training=trainable)
        lrelu5 = tf.nn.leaky_relu(bn5)

        flattened = tf.layers.flatten(lrelu5)
        logits = tf.layers.dense(flattened, 1)
        probability = tf.sigmoid(logits)
    
    if not reuse:
        print(x)
        print(lrelu1)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(lrelu5)
        print(flattened)
        print(probability)

    return probability, logits