import tensorflow as tf


def GAN_Loss(D_logits_real, D_logits_fake, label_smoothing):
    # Discriminator
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real,
                                                labels=tf.ones_like(D_logits_real) * label_smoothing))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.zeros_like(D_logits_fake)))
    D_loss = D_loss_real + D_loss_fake

    # Generator
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.ones_like(D_logits_fake)))

    return D_loss, G_loss


def GAN_Optimizer(D_loss, G_loss, lr, beta1):
    all_vars = tf.trainable_variables()
    D_vars = [var for var in all_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in all_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
        D_optimizer = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(D_loss, var_list=D_vars)
        G_optimizer = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(G_loss, var_list=G_vars)

    return D_optimizer, G_optimizer


def WGAN_Loss(D_logits_real, D_logits_fake):
    D_loss = tf.reduce_mean(D_logits_real) - tf.reduce_mean(D_logits_fake)
    G_loss = - tf.reduce_mean(D_logits_fake)

    return D_loss, G_loss


def WGAN_Optimizer(D_loss, G_loss, lr, c):
    all_vars = tf.trainable_variables()
    D_vars = [var for var in all_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in all_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(-D_loss, var_list=D_vars)
        G_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(G_loss, var_list=G_vars)

    clip_D = [p.assign(tf.clip_by_value(p, -c, c)) for p in D_vars]

    return D_optimizer, G_optimizer, clip_D