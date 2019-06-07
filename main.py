import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.datasets import mnist

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

def normalize(vector, a, b):
    assert(a < b)
    max_val = np.max(vector)
    min_val = np.min(vector)

    result = (b - a) * ( (vector - min_val) / (max_val - min_val) ) + a

    return result

(X_train, Y_train), _ = mnist.load_data()

# Expand to have 1 channel (grey images)
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)

# Print max and min and normalize
print("MAX : " + str(X_train.max()) + " and MIN: " + str(X_train.min()))
X_train = normalize(X_train, -1, 1)
print("Normalized MAX : " + str(X_train.max()) + " and MIN: " + str(X_train.min()))

assert(X_train.shape[1] == X_train.shape[2])

# Parameters network and training
epochs = 500
batchSize = 32
lr = 1e-4
Z_dim = 100
mu, sigma = 0, 1

# Sample noise for generator
def sample_Z(batch_size, img_size, mu, sigma):
    return np.random.normal(mu, sigma, size=[batch_size, img_size])
    #return np.random.uniform(-1., 1., size=[batch_size, img_size])


def generator(z, reuse=False):
    
    with tf.variable_scope("generator", reuse=reuse):        
        fc1 = tf.layers.dense(inputs=z, units=256, activation=tf.nn.leaky_relu)
        fc2 = tf.layers.dense(inputs=fc1, units=784, activation=None)
        logits = tf.nn.tanh(fc2)
        output = tf.reshape(logits, shape=[-1, 28, 28, 1])
        
        print("Generator:")
        print(z)
        print(fc1)
        print(fc2)
        print(logits)
        print(output)
        
    return output


def discriminator(x, reuse=False):
    
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.layers.Flatten()(x)
        fc1 = tf.layers.dense(inputs=x, units=256, activation=tf.nn.leaky_relu)
        fc2 = tf.layers.dense(inputs=fc1, units=1, activation=None)
        prob = tf.nn.sigmoid(fc2)
        
        if not reuse:
            print("\nDiscriminator:")
            print(x)
            print(fc1)
            print(fc2)
            print(prob)
        
    return prob


tf.reset_default_graph()

# Generator noise input
Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')

# Discriminator input
X = tf.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2], X_train.shape[3]], name='X')

# Networks
G_z = generator(Z)

D_real = discriminator(X)
D_fake = discriminator(G_z, reuse=True)


# Losses have minus sign because I have to maximize them
D_loss = - tf.reduce_mean( tf.log(D_real) + tf.log(1. - D_fake) )
G_loss = - tf.reduce_mean( tf.log(D_fake) )


# Optimizers
D_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
D_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=D_var)

G_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
G_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_var)


def saveImages(images, globalStep, samples=9):    
    size = np.sqrt(samples)
    fig = plt.figure(figsize=(15, 15))

    for i in range(np.minimum(samples, len(images))):
        plt.subplot(size, size, i+1)
        plt.imshow(np.squeeze(images[i]), cmap='gray')
        plt.axis('off')

    plt.savefig('./images/image_' + str(globalStep) + '.png')
    #plt.show()
    plt.close(fig)


tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)

# MERGE SUMMARIES - Merge all summaries into a single op
merged_summ = tf.summary.merge_all()

# VISUALIZE => tensorboard --logdir=.
summaries_dir = "./checkpoints"


indices = list(range(len(X_train)))

with tf.Session() as sess:

    # Run the initializer
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())
    
    # Epochs
    globalStep = 1
    for i in range(epochs):
        
        # Shuffle dataset every epoch
        print("Epoch " + str(i))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        
        for j in range(0, len(X_train), batchSize):
            
            # Sample noise
            noise = sample_Z(len(X_train[j:j+batchSize]), Z_dim, mu, sigma)
            
            _ = sess.run(D_optimizer, feed_dict={ X: X_train[j:j+batchSize], 
                                                  Z: noise })

            _, summary = sess.run([G_optimizer, merged_summ], feed_dict={ X: X_train[j:j+batchSize],
                                                                          Z: noise })
            summary_writer.add_summary(summary, globalStep)
            
            globalStep += 1
        
        # Check results every epoch
        save_path = saver.save(sess, "./checkpoints/model.ckpt")                
        output = sess.run(G_z, feed_dict={ Z: noise })
        saveImages(output, globalStep)