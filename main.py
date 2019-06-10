import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2
import matplotlib.image as mpimg
import pandas as pd
import os
import networks

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#base_path = "/home/francesco/UQ/Job/Tumour_GAN/"
base_path = "/scratch/cai/CANCER_GAN/"

def normalize(vector, a, b, max_val=255, min_val=0):
    assert(a < b)

    #result = (b - a) * ( (vector - min_val) / (max_val - min_val) ) + a
    result = (vector - 127.5) / 127.5
    
    return result


def getData(path, value="mel", resize=None):
    DF = pd.read_pickle(path)
    assert(len(DF["image"]) == len(DF["id"]))
    
    X = []
    for i in range(len(DF["image"])):
        
        if DF["id"][i] == value:
            if resize is None:
                X.append(DF["image"][i])
            else:
                tmp = cv2.resize(DF["image"][i], (int(resize),int(resize)), interpolation = cv2.INTER_CUBIC)
                X.append(tmp)
                
    return np.array(X)

# Load data
X1 = getData(base_path + "data/NvAndMelTest.pkl", value="nv", resize=64)
X2 = getData(base_path + "data/NvAndMelTest.pkl", value="nv", resize=64)
X_train = np.concatenate((X1, X2), axis=0)
print(X_train.shape)

# Print max and min and normalize
print("MAX : " + str(X_train.max()) + " and MIN: " + str(X_train.min()))
X_train = normalize(X_train, -1, 1)
print("Normalized MAX : " + str(X_train.max()) + " and MIN: " + str(X_train.min()))

assert(X_train.shape[1] == X_train.shape[2])

# Parameters network and training
epochs = 1000
batchSize = 128
lr = 0.0002
Z_dim = 100
mu, sigma = 0, 1


tf.reset_default_graph()

# Inputs
Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')
X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2], X_train.shape[3]], name='X')
isTrain = tf.placeholder(dtype=tf.bool)

# Networks
G_z = networks.generator(Z, isTrain)
_, D_logits_real = networks.discriminator(X, isTrain)
_, D_logits_fake = networks.discriminator(G_z, isTrain, reuse=True)


def discriminatorLoss(D_logits_real, D_logits_fake, label_smoothing=1):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real,
                                                labels=tf.ones_like(D_logits_real) * label_smoothing))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.zeros_like(D_logits_fake)))
    D_loss = D_loss_real + D_loss_fake
    
    return D_loss

def generatorLoss(D_logits_fake):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.ones_like(D_logits_fake)))
    return G_loss

D_loss = discriminatorLoss(D_logits_real, D_logits_fake, 0.9)
G_loss = generatorLoss(D_logits_fake)

# Losses have minus sign because I have to maximize them
#D_loss = - tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
#G_loss = - tf.reduce_mean(tf.log(D_fake))

all_vars = tf.trainable_variables()
D_vars = [var for var in all_vars if var.name.startswith('discriminator')]
G_vars = [var for var in all_vars if var.name.startswith('generator')]

# Optimize
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): 
    D_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)


tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)

# MERGE SUMMARIES - Merge all summaries into a single op
merged_summ = tf.summary.merge_all()

# VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + "checkpoints"

# Test noise
testNoise = networks.sample_noise(3, Z_dim, mu, sigma)
print(testNoise.shape)

def saveImages(images, epoch):    
    for i in range(len(images)):
        mpimg.imsave(base_path + "images/out-" + str(epoch) + "-" + str(i) + ".png",  ( (images[i] * 127.5) + 127.5 ).astype(np.uint8) )


indices = list(range(len(X_train)))
with tf.Session() as sess:

    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())
    
    globalStep = 0
    for i in range(epochs):        
        # Shuffle dataset every epoch
        print("Epoch " + str(i))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        
        for j in range(0, len(X_train), batchSize):
            noise = networks.sample_noise(len(X_train[j:j+batchSize]), Z_dim, mu, sigma)

            _ = sess.run(D_optimizer, feed_dict={ X: X_train[j:j+batchSize], 
                                                  Z: noise,
                                                  isTrain: True})

            noise = networks.sample_noise(len(X_train[j:j+batchSize]), Z_dim, mu, sigma)
            _, summary = sess.run([G_optimizer, merged_summ], feed_dict={ X: X_train[j:j+batchSize],
                                                                          Z: noise,
                                                                          isTrain: True })
            globalStep+=1
            summary_writer.add_summary(summary, globalStep)

        # Check results every epoch
        save_path = saver.save(sess, base_path + "checkpoints/model-" + str(i) + ".ckpt")
        G_output = sess.run(G_z, feed_dict={ Z: testNoise,
                                            isTrain: False })
        saveImages(G_output, i)