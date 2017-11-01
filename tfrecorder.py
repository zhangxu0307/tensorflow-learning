import tensorflow as tf
import numpy as np
from PIL import Image

import os

cwd = os.getcwd()

root = cwd+"/data/face_data"

TFwriter = tf.python_io.TFRecordWriter("./data/faceTF.tfrecords")

for className in os.listdir(root):
    label = int(className[1:])
    classPath = root+"/"+className+"/"
    for parent, dirnames, filenames in os.walk(classPath):
        for filename in filenames:
            imgPath = classPath+"/"+filename
            print (imgPath)
            img = Image.open(imgPath)
            print (img.size,img.mode)
            imgRaw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
                "img":tf.train.Feature(bytes_list = tf.train.BytesList(value=[imgRaw]))
            }) )
            TFwriter.write(example.SerializeToString())

TFwriter.close()




fileNameQue = tf.train.string_input_producer(["./data/faceTF.tfrecords"])
reader = tf.TFRecordReader()
key,value = reader.read(fileNameQue)
features = tf.parse_single_example(value,features={ 'label': tf.FixedLenFeature([], tf.int64),
                                           'img' : tf.FixedLenFeature([], tf.string),})

img = tf.decode_raw(features["img"], tf.uint8)
#img = tf.reshape(img, [92,112])
label = tf.cast(features["label"], tf.int32)

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        imgArr = sess.run(img)
        print (imgArr)

    coord.request_stop()
    coord.join(threads)
