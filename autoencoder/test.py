import tensorflow as tf
import data_util
import glob


t, d = data_util.get_autoencoder_dataset()

for i in t.take(1):
    print(i)
    break