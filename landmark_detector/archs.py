import tensorflow as tf
from tensorflow import keras


class D_net_gauss(keras.Model):
    def __init__(self, z_dim, N=1000):
        super(D_net_gauss, self).__init__()
        self.lin1 = keras.layers.Dense(N)
        self.lin2 = keras.layers.Dense(N)
        self.lin3 = keras.layer.Dense(1)