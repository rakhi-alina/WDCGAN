import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
import networks

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#base_path = "/home/francesco/UQ/Job/Tumour_GAN/"
base_path = "/scratch/cai/CANCER_GAN/"


# Params
Z_dim = 100
mu, sigma = 0, 1

tf.reset_default_graph()

Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')
isTrain = tf.placeholder(dtype=tf.bool)
G_z = networks.generator(Z, isTrain)


with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	for i in range(1000):
		saver.restore(sess, base_path + "checkpoints/model-" + str(i) + ".ckpt") 

		for j in range(4):
			noise = networks.sample_noise(1, Z_dim, mu, sigma)    
			G_output = sess.run(G_z, feed_dict={ Z: noise, isTrain: False })
			mpimg.imsave(base_path + "images/result-" + str(i) + "-" + str(j) + ".png",  ( (G_output[0] * 127.5) + 127.5 ).astype(np.uint8) )