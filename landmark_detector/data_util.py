import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


@tf.function
def full_tfrecord_reader(example):
    feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                           "class": tf.io.VarLenFeature(dtype=tf.int64),
                           "bb_box": tf.io.VarLenFeature(dtype=tf.float32),
                           "landmark": tf.io.VarLenFeature(dtype=tf.float32)}

    example = tf.io.parse_single_example(example, feature_description)
    image_raw = tf.sparse.to_dense(example["image"])[0]
    image = tf.io.decode_png(image_raw, channels=3)
    image_class = tf.sparse.to_dense(example["class"])
    image_bb_box = tf.sparse.to_dense(example["bb_box"])
    image_landmark = tf.sparse.to_dense(example["landmark"])

    label = tf.concat([tf.cast(image_class, dtype=tf.float32), image_bb_box, image_landmark], axis=0)
    return image, label


@tf.function
def landmark_tfrecord_reader(example):
    feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                           "class": tf.io.VarLenFeature(dtype=tf.int64),
                           "bb_box": tf.io.VarLenFeature(dtype=tf.float32),
                           "landmark": tf.io.VarLenFeature(dtype=tf.float32)}

    example = tf.io.parse_single_example(example, feature_description)
    image_raw = tf.sparse.to_dense(example["image"])[0]
    image = tf.io.decode_png(image_raw, channels=3)
    image_class = tf.sparse.to_dense(example["class"])
    image_bb_box = tf.sparse.to_dense(example["bb_box"])
    image_landmark = tf.sparse.to_dense(example["landmark"])

    return image, image_landmark


@tf.function
def class_landmark_tfrecord_reader(example):
    feature_description = {"image": tf.io.VarLenFeature(dtype=tf.string),
                           "class": tf.io.VarLenFeature(dtype=tf.int64),
                           "bb_box": tf.io.VarLenFeature(dtype=tf.float32),
                           "landmark": tf.io.VarLenFeature(dtype=tf.float32)}

    example = tf.io.parse_single_example(example, feature_description)
    image_raw = tf.sparse.to_dense(example["image"])[0]
    image = tf.io.decode_png(image_raw, channels=3)
    image_class = tf.sparse.to_dense(example["class"])
    image_bb_box = tf.sparse.to_dense(example["bb_box"])
    image_landmark = tf.sparse.to_dense(example["landmark"])

    label = tf.concat([tf.cast(image_class, dtype=tf.float32), image_landmark], axis=0)

    return image, label


# train_type: label의 형태를 결정.
# data_path: tfrecord 파일이 있는 위치
# batch_size: batch size
def get_cu_dataset(train_type='landmark', data_path='./data/cu_dataset.tfrecord', batch_size=32):
    # landmark 만 label 되어있는 데이터 셋
    if train_type == 'landmark':
        ds = tf.data.TFRecordDataset(data_path).map(landmark_tfrecord_reader)
        ds = ds.map(image_scaling).batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # class와 함께 label 되어있는 데이터셋. class, landmark 순서
    elif train_type == 'class':
        ds = tf.data.TFRecordDataset(data_path).map(class_landmark_tfrecord_reader)
        ds = ds.map(image_scaling).batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # bb-box, class, landmark 모두 label. class, bb-box, landmark 순서
    else:
        ds = tf.data.TFRecordDataset(data_path).map(full_tfrecord_reader)
        ds = ds.map(image_scaling).batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def image_resize(x, y):
    x = tf.image.resize(x, size=[224, 224])
    return x, y


def image_scaling(x, y):
    x = preprocess_input(tf.cast(x, dtype=tf.float32))
    y /= 224.
    return x, y

