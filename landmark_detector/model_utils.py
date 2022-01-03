import tensorflow as tf
from tensorflow import keras


def builder(net, input_size):
    # Q build
    Q_sample = tf.random.normal(shape=(1, input_size, input_size, 3))
    Q_out = net.Q(Q_sample)

    # P build
    P_sample = tf.random.normal(shape=Q_out.shape)
    P_out = net.P(P_sample)

    # D_z build
    D_z_sample = tf.random.normal(shape=Q_sample.shape)
    D_z_out = net.D_z(D_z_sample)

    # D build
    D_sample = tf.random.normal(shape=P_out.shape)
    D_out = net.D(D_sample)

    return net
