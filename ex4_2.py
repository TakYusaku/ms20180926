#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf

import common

class NN:
    def __init__(self):
        ## ロジスティック回帰モデルの設定
        self.dim_input=1   # 入力ベクトル次元
        self.dim_output=1  # 出力ベクトル次元
        self.num_neuron = 10
        self.lam = 0.005

        # define weight
        input_weight = tf.Variable(tf.random_uniform([self.dim_input, self.num_neuron]), name="input_weight")
        w10 = tf.Variable(tf.zeros([self.num_neuron]), name="bias_input")
        output_weight = tf.Variable(tf.random_uniform([self.num_neuron, self.dim_output]), name="output_weight")
        w20 = tf.Variable(tf.zeros([self.dim_output]), name="bias_output")

        # 入力xと教師データt
        self.x = tf.placeholder(tf.float32, [None, self.dim_input], name="x-input")
        self.t = tf.placeholder(tf.float32, [None, self.dim_output], name="t-teacher")

        # (hiddenunit)neuron input ha
        self.ha = tf.matmul(self.x, input_weight) + w10

        # activation dim[none x num_neuron]
        self.z = tf.nn.sigmoid(self.ha, name="hiddenunit_output")

        # (outputunit) neuron input y
        self.y = tf.matmul(self.z, output_weight) + w20  # 回帰問題なので恒等写像
        """
        # 誤差は二乗誤差
        self.error = tf.reduce_mean(tf.square(self.y - self.t))
        """

        # 正則化項付きの2乗誤差
        penalty_term = self.lam * (tf.nn.l2_loss(input_weight) + tf.nn.l2_loss(output_weight))
        self.error = tf.reduce_mean(tf.square(self.y - self.t)) + penalty_term

        # 学習の1ステップで誤差を最小化する
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.error)

        ## sessはtensorflowの学習セッション
        self.sess = tf.InteractiveSession()

        # すべてのパラメータを初期化する(乱数ベース)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, data_x, data_t, max_epoch):
        ## 学習する
        errs = []
        for i in range(max_epoch):
#            print(self.sess.run(self.ha, feed_dict={self.x:data_x}))
            err,train = self.sess.run([self.error, self.train_step], feed_dict={self.x:data_x, self.t:data_t})
            errs.append(err)
            if i%100==0:
                print ('error(cross entropy) is %f'%(err)),

        print ('Done training')
        return errs

    def predict(self, input):
        return self.probability(input)

    def probability(self, input):
        return self.sess.run(self.y, feed_dict={self.x:input})

    def accuracy_score(self, x, y):
        pred = self.predict(x)
        return accuracy_score(pred, y)

def plot_learn(errs):
    ## 学習過程を表示
    max_epoch = len(errs)
    plt.figure(2)
    plt.plot(np.arange(0,max_epoch), errs, 'k')
    plt.xlabel('epoch')
    plt.ylabel('error(cross entropy)')

def plot_eval(model, data_x, data_t):
    N = 1000
    xs = np.linspace(0, 1, N)
    pred=model.predict( np.resize(xs, (len(xs),1)) )
    plt.figure(1)
    plt.plot(xs, pred, 'r-')
    plt.plot(data_x, data_t, 'bo')

def main():
    ## 学習データの準備
    filename = 'data.csv'
    tmp = np.loadtxt(filename, delimiter=',')
    train_x = np.resize(tmp[:,0],(len(tmp[:,0]),1))
    train_y = np.resize(tmp[:,1],(len(tmp[:,1]),1))

    ## モデル
    model = NN()

    ## 学習
    max_epoch=10000
    errs = model.fit(train_x, train_y, max_epoch)

    ## 学習過程の表示
    plot_learn(errs)

    ## 学習結果(予測と学習データ)の表示
    plot_eval(model, train_x, train_y)

    plt.show()


if __name__ == '__main__':
    main()
