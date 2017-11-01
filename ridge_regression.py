import tensorflow as tf
import numpy as np
import pandas as pd

train = pd.read_csv("./data/dataset1-a9a-training.txt", encoding="utf-8")
print(train.describe())

columnNum = train.values.shape[1] # 获取数据的列数，其中最后一列是label

# 使用pandas获取全部数据，评估效果用
def getAll(filename):

    data = pd.read_csv(filename, dtype=np.float32)
    example = data.values[:,:-1]
    label = data.values[:,-1]
    return example, label

# tf获取数据
def read_data(filenameQueue):
    reader = tf.TextLineReader()
    key, value = reader.read(filenameQueue)

    recordDefaults = []
    for i in range(columnNum):
        recordDefaults.append([0.0])
    col = tf.decode_csv(value, record_defaults=recordDefaults)
    features = tf.squeeze(tf.reshape(tf.stack([col[:-1]]), [columnNum-1, 1]), squeeze_dims=1) # 统一格式为[n_samples,1]
    label = col[-1]
    return features, label

# tf分批数据
def input_batch(filename, batchSize, dequeue = 10000):

    fileNameQue = tf.train.string_input_producer([filename], shuffle=True)
    example, label = read_data(fileNameQue)
    min_after_dequeue = dequeue   # 样本池调整的大一些随机效果好
    capacity = min_after_dequeue + 3 * batchSize
    exampleBatch, labelBatch = tf.train.shuffle_batch([example, label], batch_size=batchSize, capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)
    return exampleBatch, labelBatch

# 岭回归计算图
def ridge_regression(exampleBatch, labelBatch, lamda = 0.5, alpha = 0.1):

    with tf.name_scope("ridge"):

        W = tf.Variable(tf.random_normal([columnNum-1, 1]), name="W")
        b = tf.Variable(tf.random_normal([1]), name="b")
        logits = tf.matmul(exampleBatch, W)+b
        loss = tf.reduce_mean(tf.square(labelBatch-logits)) + lamda*tf.norm(W)
        train = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
        return train, loss, W, b

# 评价函数，计算准确率
def evaluate(W, b, testData, testLabel):

    testExample = tf.Variable(testData, trainable=False)
    tfLabel = tf.Variable(testLabel, trainable=False)
    tfLabel = tf.equal(tfLabel, 1)
    tfLabel = tf.reshape(tfLabel, [-1, 1])
    pred = tf.matmul(testExample, W)+b
    res = tf.equal(tf.greater(pred, 0.0), tfLabel) # 以0为分界点分类
    acc = tf.reduce_mean((tf.cast(res, dtype=tf.float32))) # 转换成浮点型，整型计算会一直结果为0
    return acc


if __name__ == "__main__":

    exampleBatch, labelBatch = input_batch("./data/dataset1-a9a-training.txt", batchSize=100)
    train, loss, W, b = ridge_regression(exampleBatch, labelBatch)
    testData, testLabel = getAll("./data/dataset1-a9a-training.txt")
    acc = evaluate(W, b, testData, testLabel)

    maxIter = 1000 # 最大迭代次数
    with tf.Session() as sess:

        init = tf.global_variables_initializer() # 初始化放在计算图后
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(maxIter):

            #example, label = sess.run([exampleBatch, labelBatch])
            #print(example.shape)
            #print(label)
            _, lossArr, accArr = sess.run([train, loss, acc])
            print(lossArr, accArr)
            #print(logits)

        coord.request_stop()
        coord.join(threads)


