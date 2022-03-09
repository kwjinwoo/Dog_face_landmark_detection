import xmltodict
import json
from tqdm import tqdm
import re
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random


def point_adjust(parts, image_width, image_height):
    target_size = 224.0
    new_points = []
    for i in range(len(parts)):
        point = parts[i]
        if (i + 1) % 2 != 0:
            new_point = float((target_size/image_width)*point)
        else:
            new_point = float((target_size / image_height) * point)
        new_points.append(new_point)

    return new_points


def class_one_hot_encoder(labels):
    labels = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels)
    return encoder


def get_class(file_path):
    class_name = file_path.split('/')[1]
    class_name = re.sub('[\d\.]', '', class_name)
    return class_name


xml_path = './data/all_dogs_labeled.xml'
f = open(xml_path).read()
merge_xml = xmltodict.parse(f)

merge_xml = json.dumps(merge_xml)
merge_xml = json.loads(merge_xml)

images = merge_xml['dataset']['images']['image']

label_list = []
for image in tqdm(images, desc='get_labels'):
    path = image['@file']
    temp_class = get_class(path)
    label_list.append(temp_class)

num_classes = len(set(label_list))
print('num_classes: ', len(set(label_list)))
encoder = class_one_hot_encoder(label_list)

# image keys: '@file', '@width', '@height', 'box'
# box keys: '@top', '@left', '@width', '@height', 'label', 'part'
# part list: 'head_top', 'lear_base', 'lear_tip', 'leye, 'nose', 'rear_base', 'rear_tip', 'reye'
# part keys: '@name', '@x', '@y'
save_path = './data/cu_dataset_yolo.tfrecord'
c = 0
random.shuffle(images)
with tf.io.TFRecordWriter(save_path) as f:
    for image in tqdm(images):
        path = image['@file']

        image_class = get_class(path)       # dog species
        image_class = encoder.transform([[image_class]])[0]
        image_class = image_class.astype('int64')

        # 깨진 파일 거르기 + 객체 2개인거
        # image
        image_file = tf.io.decode_image(tf.io.read_file('./data/' + path)).numpy()
        image_file = tf.image.resize(image_file, size=[256, 256])
        image_file = tf.io.encode_jpeg(tf.cast(image_file, dtype=tf.uint8)).numpy()
        image_width = int(image['@width'])
        image_height = int(image['@height'])

        image_box = image['box']
        try:
            top = int(image_box['@top'])
            left = int(image_box['@left'])
            width = int(image_box['@width'])
            height = int(image_box['@height'])
            bb_box_list = [top, left, width, height]
            bb_box_list = point_adjust(bb_box_list, image_width, image_height)

            # 'head_top', 'lear_base', 'lear_tip', 'leye, 'nose', 'rear_base', 'rear_tip', 'reye'
            # x,y 순서
            landmark_list = []
            for point in image_box['part']:
                x = int(point['@x'])
                y = int(point['@y'])
                landmark_list.extend([x, y])
            landmark_list = point_adjust(landmark_list, image_width, image_height)

            # record
            record = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file])),
                        'class': tf.train.Feature(int64_list=tf.train.Int64List(value=image_class)),
                        'bb_box': tf.train.Feature(float_list=tf.train.FloatList(value=bb_box_list)),
                        'landmark': tf.train.Feature(float_list=tf.train.FloatList(value=landmark_list))
                    }
                )
            )
            f.write(record.SerializeToString())
            c += 1
        except TypeError as e:



print('total saved_image: ', c)

