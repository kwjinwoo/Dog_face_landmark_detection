import tensorflow as tf
from tensorflow import keras


eps = 1e-8


# reduction은 mean
def loss_recon(x, x_recon):
    diff = tf.math.abs(x - x_recon) * 255.
    diff = tf.reshape(diff, (diff.shape[0], -1))
    mean_diff = tf.reduce_mean(diff, axis=1)
    return tf.reduce_mean(mean_diff)


# ssim loss의 평균
def loss_struct(x, x_recon):
    cs = tf.image.ssim(x, x_recon, 1)
    loss = 1 - cs
    return tf.reduce_mean(loss)


def loss_enc(z_sample):
    z_real = tf.random.normal(shape=z_sample.shape)


def loss_adv(x):
    None


def loss_D_Z(D_real, D_fake):
    loss = tf.math.log(D_real + eps) + tf.math.log(1 - D_fake + eps)
    return -tf.reduce_mean(loss)