import numpy as np
import tensorflow as tf
import os
import networks, util, loss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#base_path = "/home/francesco/UQ/Job/Tumour_GAN/"
base_path = "/scratch/cai/CANCER_GAN/"

# Parameters
epochs = 1000
batchSize = 128
lr = 0.0002
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
dataset = tf.data.Dataset.from_tensor_slices(X_train).repeat(epochs).shuffle(buffer_size=10000).batch(batchSize)
iterator = dataset.make_one_shot_iterator()
X = iterator.get_next()
Z = tf.placeholder(tf.float32, shape=[None, 1, 1, Z_dim], name='Z')
isTrain = tf.placeholder(dtype=tf.bool)

# Networks
G_z = networks.generator(Z, isTrain)
_, D_logits_real = networks.discriminator(X, isTrain)
_, D_logits_fake = networks.discriminator(G_z, isTrain, reuse=True)

# Losses and optimizer
D_loss = loss.discriminatorLoss(D_logits_real, D_logits_fake, 1.0)
G_loss = loss.generatorLoss(D_logits_fake)
D_optimizer, G_optimizer = loss.getOptimizers(lr, beta1, D_loss, G_loss)

# Tensorboard
summaries_dir = base_path + "checkpoints"
merged_summ = util.tensorboard(summaries_dir, D_loss, G_loss)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())
    
    globalStep = 0
    noise_shape = [batchSize, 1, 1, Z_dim]
    try: 
        while True:
            noise = networks.sample_noise(noise_shape, mu, sigma)
            sess.run(D_optimizer, feed_dict={ isTrain: True, Z: noise } )

            noise = networks.sample_noise(noise_shape, mu, sigma)
            _, summary = sess.run([G_optimizer, merged_summ], feed_dict={ Z: noise, isTrain: True })
            summary_writer.add_summary(summary, globalStep)

            # Save checkpoints and images
            if globalStep % 100 == 0:
                save_path = saver.save(sess, base_path + "checkpoints/model-" + str(globalStep) + ".ckpt")
                G_output = sess.run(G_z, feed_dict={ Z: networks.sample_noise([4, 1, 1, Z_dim], mu, sigma), isTrain: False })
                util.saveImages(base_path + "images/out-" + str(globalStep), G_output)
            globalStep += 1
    except tf.errors.OutOfRangeError:
        pass