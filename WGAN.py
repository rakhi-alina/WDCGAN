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
                
    return np.array(X, dtype=np.float32)

# Load data
X1 = getData(base_path + "data/NvAndMelTrain.pkl", value="nv", resize=64)
X2 = getData(base_path + "data/NvAndMelTest.pkl", value="nv", resize=64)
X_train = np.concatenate((X1, X2), axis=0)
assert(X_train.shape[1] == X_train.shape[2])
print(X_train.shape)

# Print max and min and normalize
print("MAX : " + str(X_train.max()) + " and MIN: " + str(X_train.min()))
X_train = normalize(X_train, -1, 1)
print("Normalized MAX : " + str(X_train.max()) + " and MIN: " + str(X_train.min()))

# Parameters network and training
epochs = 1000
batchSize = 128
lr = 5e-5
c = 1e-2
n_critic = 5
Z_dim = 100
mu, sigma = 0, 1

# Tensorflow
tf.reset_default_graph()

# Inputs
dataset = tf.data.Dataset.from_tensor_slices(X_train).repeat(epochs).shuffle(buffer_size=1000).batch(batchSize)
iterator = dataset.make_one_shot_iterator()
X = iterator.get_next()
Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')
isTrain = tf.placeholder(dtype=tf.bool)

# Networks
G_z = networks.generator(Z, isTrain)
_, D_logits_real = networks.discriminator(X, isTrain)
_, D_logits_fake = networks.discriminator(G_z, isTrain, reuse=True)

# Losses have minus sign because I have to maximize them
G_loss = - tf.reduce_mean(D_logits_fake)
D_loss = tf.reduce_mean(D_logits_real) - tf.reduce_mean(D_logits_fake)

D_weights = [w for w in tf.global_variables() if 'discriminator' in w.name]
G_weights = [w for w in tf.global_variables() if 'generator' in w.name]

all_vars = tf.trainable_variables()
D_vars = [var for var in all_vars if var.name.startswith('discriminator')]
G_vars = [var for var in all_vars if var.name.startswith('generator')]

# Optimize
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): 
    D_optimizer = (tf.train.RMSPropOptimizer(learning_rate=lr).minimize(-D_loss, var_list=D_vars))
    G_optimizer = (tf.train.RMSPropOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_vars))

# clip the weights, so that they fall in [-c, c]
clip_updates = [w.assign(tf.clip_by_value(w, -c, c)) for w in D_weights]

# Summaries | VISUALIZE => tensorboard --logdir=.
summaries_dir = base_path + "checkpoints"
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)
merged_summ = tf.summary.merge_all()

def saveImages(images, epoch):    
    for i in range(len(images)):
        mpimg.imsave(base_path + "images/out-" + str(epoch) + "-" + str(i) + ".png",  ( (images[i] * 127.5) + 127.5 ).astype(np.uint8) )


with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())
    
    globalStep = 0
    try: 
        while True:
            # Train Discriminator more than Generator
            n_critic_epoch = 100 if globalStep < 25 or (globalStep+1) % 500 == 0 else n_critic

            # Train Discriminator
            for i in range(n_critic_epoch):
                sess.run(clip_updates)
                noise = networks.sample_noise(batchSize, Z_dim, mu, sigma)
                sess.run(D_optimizer, feed_dict={ isTrain: True, Z: noise } )

            # Train Generator
            noise = networks.sample_noise(batchSize, Z_dim, mu, sigma)
            _, summary = sess.run([G_optimizer, merged_summ], feed_dict={ Z: noise, isTrain: True })
            globalStep += 1
            summary_writer.add_summary(summary, globalStep)

            # Save checkpoints and images
            if globalStep % 100 == 0:
                save_path = saver.save(sess, base_path + "checkpoints/model-" + str(globalStep) + ".ckpt")
                G_output = sess.run(G_z, feed_dict={ Z: networks.sample_noise(3, Z_dim, mu, sigma), isTrain: False })
                saveImages(G_output, globalStep)

    except tf.errors.OutOfRangeError:
        pass