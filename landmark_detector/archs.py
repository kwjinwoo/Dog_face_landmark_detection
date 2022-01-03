import tensorflow as tf
from tensorflow import keras


class D_net_gauss(keras.Model):
    def __init__(self, z_dim, N=1000):
        super(D_net_gauss, self).__init__()
        self.lin1 = keras.layers.Dense(N)
        self.lin2 = keras.layers.Dense(N)
        self.lin3 = keras.layers.Dense(1)
        self.dropout = keras.layers.Dropout(0.2)
        self.relu = keras.layers.ReLU()
        self.sigmoid = keras.layers.Activation('sigmoid')

    def call(self, x):
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dropout(x)
        x= self.relu(x)

        return self.sigmoid(self.lin3(x))


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 32
        nc = 3
        self.model = [
            keras.layers.Conv2D(ndf, 4, strides=2, padding='same', use_bias=False),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(ndf * 2, 4, strides=2, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(ndf * 4, 4, strides=2, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(ndf * 8, 4, strides=2, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(1, 4, strides=1, padding='valid', use_bias=False),
            keras.layers.GlobalAvgPool2D(),
            keras.layers.Activation('sigmoid')
        ]
        self.model = keras.Sequential(self.model)

    def call(self, x):
        x = self.model(x)
        return tf.squeeze(x, 1)
