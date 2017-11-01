import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 200 # 样本点数目
x = np.linspace(-1, 1, N)
y = 2.0*x + np.random.standard_normal(x.shape)*0.3+0.5 # 生成线性数据
x = x.reshape([N, 1]) # 转换一下格式，准备feed进placeholder
y = y.reshape([N, 1])

plt.scatter(x, y)
plt.plot(x, 2*x+0.5)
plt.show()

# 建图
inputx = tf.placeholder(dtype=tf.float32, shape=[None, 1])
groundY = tf.placeholder(dtype=tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([1, 1], stddev=0.01))
b = tf.Variable(tf.random_normal([1], stddev=0.01))
pred = tf.matmul(inputx, W)+b
loss = tf.reduce_sum(tf.pow(pred-groundY, 2))

# 优化目标函数
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 加入监控点
tf.summary.scalar("loss", loss)
merged = tf.summary.merge_all()

# 初始化所有变量
init = tf.global_variables_initializer()



with tf.Session() as sess:

    # 定义日志文件
    writer = tf.summary.FileWriter("./log/", sess.graph)

    sess.run(init)

    for i in range(20):
        sess.run(train,feed_dict={inputx:x, groundY:y})
        predArr, lossArr = sess.run([pred, loss], feed_dict={inputx:x, groundY:y})
        print(lossArr)

        summary_str = sess.run(merged, feed_dict={inputx:x, groundY:y})
        writer.add_summary(summary_str, i) # 向日志文件写入监控点数据

        # 作图观察
        WArr, bArr = sess.run([W, b])
        print(WArr, bArr)
        plt.scatter(x, y)
        plt.plot(x, WArr * x + bArr)
        plt.show()

