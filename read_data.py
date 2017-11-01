import tensorflow as tf

def read_data(fileNameQue):

    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(value, features={'label': tf.FixedLenFeature([], tf.int64),
                                                        'img': tf.FixedLenFeature([], tf.string),})
    img = tf.decode_raw(features["img"], tf.uint8)
    img = tf.reshape(img, [92,112]) # 恢复图像原始大小
    label = tf.cast(features["label"], tf.int32)

    return img, label

def batch_input(filename, batchSize):

    fileNameQue = tf.train.string_input_producer([filename], shuffle=True)
    img, label = read_data(fileNameQue) # fetch图像和label
    min_after_dequeue = 1000
    capacity = min_after_dequeue+3*batchSize
    # 预取图像和label并随机打乱，组成batch，此时tensor rank发生了变化，多了一个batch大小的维度
    exampleBatch,labelBatch = tf.train.shuffle_batch([img, label],batch_size=batchSize, capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue)
    return exampleBatch,labelBatch

if __name__ == "__main__":

    init = tf.initialize_all_variables()
    exampleBatch, labelBatch = batch_input("./data/faceTF.tfrecords", batchSize=10)

    with tf.Session() as sess:

        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(100):
            example, label = sess.run([exampleBatch, labelBatch])
            print(example.shape)

        coord.request_stop()
        coord.join(threads)






