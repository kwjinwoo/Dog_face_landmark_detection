import tensorflow as tf
import glob
from tqdm import tqdm


images_path = glob.glob('./data/images/*/*.jpg')
print('total images: ', len(images_path))

in_file = 5000
num_iter = (len(images_path) // in_file) + 1
print('the number of tfrec files: ', num_iter)
print()
total_save_images = 0

for i in range(num_iter):
    sub_path = images_path[in_file * i:in_file * (i + 1)]
    save_path = 'data/autoencoder/ae_' + str(i) + '.tfrec'
    print('write ' + str(i) + ' start...')

    with tf.io.TFRecordWriter(save_path) as f:
        c = 0
        for path in tqdm(sub_path):
            image = open(path, 'rb').read()
            record = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                    }
                )
            )
            f.write(record.SerializeToString())
            c += 1
    print('write ' + str(i) + ' end / images in file: ', c)
    print()
    total_save_images += c

    print('total_save_images: ', total_save_images)
