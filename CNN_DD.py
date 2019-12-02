import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class ConvModel(object):
    def __init__(self, lr, batch_size, iter_num):
        self.lr = lr
        self.batch_size = batch_size
        self.iter_num = iter_num

        self.X_flat = tf.placeholder(tf.float32, [None, 784])
        self.X = tf.reshape(self.X_flat, [-1, 28, 28, 1])  # 本次要用卷积进行运算，所以使用2维矩阵。从这个角度讲，利用了更多的位置信息。
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.dropRate = tf.placeholder(tf.float32)

        conv1 = tf.layers.conv2d(self.X, 32, 5, padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=0),
                                 bias_initializer=tf.constant_initializer(0.1))
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 5, padding='same', activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=0),
                                 bias_initializer=tf.constant_initializer(0.1))
        pool1 = tf.layers.max_pooling2d(conv2, 2, 2)
        flatten = tf.reshape(pool1, [-1, 7 * 7 * 64])
        dense1 = tf.layers.dense(flatten, 1024, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=0),
                                 bias_initializer=tf.constant_initializer(0.1))
        dense1_ = tf.nn.dropout(dense1, self.dropRate)
        dense2 = tf.layers.dense(dense1_, 10, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=0),
                                 bias_initializer=tf.constant_initializer(0.1))

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=dense2)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        # 用于模型训练
        self.correct_prediction = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(dense2, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # 用于保存训练好的模型
        self.saver = tf.train.Saver()

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 先初始化所有变量。
            for i in range(self.iter_num):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)  # 读取一批数据
                loss, _ = sess.run([self.loss, self.train_step],
                                   feed_dict={self.X_flat: batch_x, self.y: batch_y,
                                              self.dropRate: 0.5})  # 每调用一次sess.run，就像拧开水管一样，所有self.loss和self.train_step涉及到的运算都会被调用一次。
                if i % 1000 == 0:
                    train_accuracy = sess.run(self.accuracy, feed_dict={self.X_flat: batch_x, self.y: batch_y,
                                                                        self.dropRate: 1.})  # 把训练集数据装填进去
                    test_x, test_y = mnist.test.next_batch(self.batch_size)
                    test_accuracy = sess.run(self.accuracy, feed_dict={self.X_flat: test_x, self.y: test_y,
                                                                       self.dropRate: 1.})  # 把测试集数据装填进去
                    print('iter\t%i\tloss\t%f\ttrain_accuracy\t%f\ttest_accuracy\t%f' % (
                    i, loss, train_accuracy, test_accuracy))
            self.saver.save(sess, 'model/mnistModel')  # 保存模型

    def test(self):
        with tf.Session() as sess:
            self.saver.restore(sess, 'model/mnistModel')
            Accuracy = []
            for i in range(int(10000 / self.batch_size)):
                test_x, test_y = mnist.test.next_batch(self.batch_size)
                test_accuracy = sess.run(self.accuracy,
                                         feed_dict={self.X_flat: test_x, self.y: test_y, self.dropRate: 1.})
                Accuracy.append(test_accuracy)
            print('==' * 15)
            print('Test Accuracy: ', np.mean(np.array(Accuracy)))


model = ConvModel(0.001, 64, 2000)  # 学习率为0.001，每批传入64张图，训练2000次
model.train()  # 训练模型
model.test()  # 预测
