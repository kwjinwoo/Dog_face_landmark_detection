import tensorflow as tf
from tensorflow import keras


# reduction은 mean
def loss_recon(x, x_recon):
    diff = tf.math.abs(x - x_recon) * 255.
    diff = tf.reshape(diff, (diff.shape[0], -1))
    mean_diff = tf.reduce_mean(diff, axis=1)
    return tf.reduce_mean(mean_diff)


# ssim loss의 평균
def loss_struct(x, x_recon):
    cs = tf.image.ssim(x, x_recon, 1)  # study 필요
    loss = 1 - cs   # ???
