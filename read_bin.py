import tensorflow as tf
import numpy as np

# 预定义图像数据信息
labelBytes = 1
witdthBytes = 32
heightBytes = 32
depthBytes = 3
imageBytes = witdthBytes*heightBytes*depthBytes
recordBytes = imageBytes+labelBytes

filename_queue = tf.train.string_input_producer(["./data/train.bin"])
reader = tf.FixedLengthRecordReader(record_bytes=recordBytes) # 按固定长度读取二进制文件
key,value = reader.read(filename_queue)

bytes = tf.decode_raw(value,out_type=tf.uint8) # 解码为uint8,0-255 8位3通道图像
label = tf.cast(tf.strided_slice(bytes,[0],[labelBytes]),tf.int32) # 分割label并转化为int32

originalImg  = tf.reshape(tf.strided_slice(bytes,[labelBytes],[labelBytes+imageBytes]),[depthBytes,heightBytes,witdthBytes])
# 分割图像，此时按照数据组织形式深度在前
img = tf.transpose(originalImg,[1,2,0]) # 调整轴的顺序，深度在后


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        imgArr = sess.run(img)
        print (imgArr.shape)

    coord.request_stop()
    coord.join(threads)
