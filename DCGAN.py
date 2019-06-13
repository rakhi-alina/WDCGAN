import numpy as np
import tensorflow as tf
import os
import network, util, loss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#base_path = "/home/francesco/UQ/Job/Tumour_GAN/"
base_path = "/scratch/cai/CANCER_GAN/"

# Parameters
epochs = 1000
batchSize = 256
lr = 1e-4
beta1 = 0.5
Z_dim = 100
mu, sigma = 0, 1

# Load and normalize data
X1 = util.getDataTumour(base_path + "data/NvAndMelTrain.pkl", value="nv", resize=64)
X2 = util.getDataTumour(base_path + "data/NvAndMelTest.pkl", value="nv", resize=64)
X_train = np.concatenate((X1, X2), axis=0)
#X_train = util.getDataTumour(base_path + "data/NvAndMelTest.pkl", value="nv", resize=64)
assert(X_train.shape[1] == X_train.shape[2])
print(X_train.shape)
X_train = util.normalize(X_train, -1, 1)

# Inputs
tf.reset_default_graph()
dataset = tf.data.Dataset.from_tensor_slices(X_train).repeat(epochs).shuffle(buffer_size=len(X_train)).batch(batchSize, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
X = iterator.get_next()
batchSizeTensor = tf.placeholder(tf.int32)
Z = network.sample_noise(batchSizeTensor, Z_dim)
isTraining = tf.placeholder(dtype=tf.bool)

# Networks
G_z = network.generator(Z, isTraining)
D_logits_real = network.discriminator(X)
D_logits_fake = network.discriminator(G_z, reuse=True)

# Losses and optimizer
D_loss, G_loss = loss.GAN_Loss(D_logits_real, D_logits_fake, 1.0)
D_optimizer, G_optimizer = loss.GAN_Optimizer(D_loss, G_loss, lr, beta1)

# Tensorboard
summaries_dir = base_path + "checkpoints"
merged_summ = util.tensorboard(summaries_dir, D_loss, G_loss)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	saver = tf.train.Saver(max_to_keep=1000)
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())
    
	globalStep = 0
	try: 
		while True:
			sess.run(D_optimizer, feed_dict={ isTraining: True, batchSizeTensor : batchSize })
			sess.run(G_optimizer, feed_dict={ isTraining: True, batchSizeTensor : batchSize })
			
			# Save summary
			'''if globalStep % 20 == 0:
				summary = sess.run(merged_summ, feed_dict={ isTraining: True, batchSizeTensor : batchSize })
				summary_writer.add_summary(summary, globalStep)'''
                
            # Save checkpoints and images
			if globalStep % 100 == 0:
				save_path = saver.save(sess, base_path + "checkpoints/model-" + str(globalStep) + ".ckpt")
				G_output = sess.run(G_z, feed_dict={ isTraining : False, batchSizeTensor : 5 })
				util.saveImages(base_path + "images/out-" + str(globalStep), G_output)
                
			globalStep += 1
	except tf.errors.OutOfRangeError:
		pass