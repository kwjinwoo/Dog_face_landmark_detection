import tensorflow as tf
from tensorflow import keras


# epochs = 100
# lr = 0.01
# batch_size = 32
# n_step = len(x) // batch_size
def train(aae, epochs, lr, dataset):
    opt_E = keras.optimizers.Adam(lr)
    opt_G = keras.optimizers.Adam(lr)
    opt_D_z = keras.optimizers.Adam(lr)
    opt_D = keras.optimizers.Adam(lr)

    for epoch in range(epochs):
        epoch += 1

        run_epoch(dataset, aae)


def run_epoch(dataset, aee):
    dataset
