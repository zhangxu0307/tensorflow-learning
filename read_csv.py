import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["./data/iris.csv", "./data/iris2.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# key返回的是读取文件和行数信息 b'./data/iris.csv:146'
# value是按行读取到的原始字符串，送到下面的decoder去解析

record_defaults = [[1.0], [1.0], [1.0], [1.0], ["Null"]] # 这里的数据类型决定了读取的数据类型，而且必须是list形式
col = tf.decode_csv(value, record_defaults=record_defaults) # 解析出的每一个属性都是rank为0的标量
features = tf.stack([col[:4]])
col5 = col[-1]

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(200):
        example, label = sess.run([features, col5])
        print (example, col5)
        print (sess.run(tf.rank(features)))

    coord.request_stop()
    coord.join(threads)