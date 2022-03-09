import tensorflow as tf
import glob


@tf.function
def tfrecord_reader(example):
    feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string)}

    example = tf.io.parse_single_example(example, feature_description)
    image_raw = tf.sparse.to_dense(example["image"])[0]
    image = tf.io.decode_png(image_raw, channels=3)
    return image


def image_resize(x):
    x = tf.image.resize(x, [224, 224])
    return x


def image_scaling(x):
    x = tf.cast(x, tf.float32) / 255.
    return x


def get_autoencoder_dataset(data_path='./data/autoencoder/*.tfrec', batch_size=32):
    data_path = glob.glob(data_path)

    train_dataset = tf.data.TFRecordDataset(data_path).map(tfrecord_reader)
    train_dataset = train_dataset.map(image_resize).map(image_scaling).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset
