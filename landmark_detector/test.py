import tensorflow as tf


a = [[1, 2, 3],
     [1, 2, 3]]
b = [[[2, 2, 2],
     [2, 2, 2]]]
b = tf.Variable(b)
print(b[:])
