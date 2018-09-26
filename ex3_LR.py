#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

import common

class LogisticRegression:
    def __init__(self):
        ## ロジスティック回帰モデルの設定
        self.dim_input=2   # 入力ベクトル次元
        self.dim_output=2  # 出力ベクトル次元

        # 入力xと教師データt
        self.x = tf.placeholder(tf.float32, [None, self.dim_input], name="x-input")
        self.t = tf.placeholder(tf.float32, [None, self.dim_output], name="t-teacher")

        # 出力y(ロジスティック回帰)
        W = tf.Variable(tf.random_uniform([self.dim_input, self.dim_output]), name="weight")
        b = tf.Variable(tf.zeros([self.dim_output]), name="bias")
        self.y = tf.nn.sigmoid(tf.matmul(self.x,W)+b, name="output")

        # 誤差は交差エントロピー関数
        self.cross_entropy=tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.t, logits=self.y))

        # 学習の1ステップで交差エントロピーを最小化する
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.cross_entropy)

        ## sessはtensorflowの学習セッション
        self.sess = tf.InteractiveSession()

        # すべてのパラメータを初期化する(乱数ベース)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, data_x, data_t):
        ## 学習する
        max_epoch=5000
        errs = []
        for i in range(max_epoch):
            err,train = self.sess.run([self.cross_entropy, self.train_step], feed_dict={self.x:data_x, self.t:data_t})
            errs.append(err)
            #if i%100==0:
                #print ('error(cross entropy) is %f'%(err)),

        print ('Done training')
        return errs

    def predict(self, input):
        pred = self.probability(input)
        pred[pred<0.5]=0
        pred[pred>=0.5]=1
        return pred

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
    common.plot_eval(model, data_x, data_t)

    # 確率を等高線表示に
    plt.figure(3)
    # # -3<x1<3, -3<x2<3の範囲でテストデータを生成
    testX,testY = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    testXin = testX.reshape(1,testX.size)
    testXin = testXin[0,:]
    testYin = testY.reshape(1,testY.size)
    testYin = testYin[0,:]
    test = np.array([testXin, testYin]).transpose()

    prob = model.probability(test)

    c1mask = data_t==[1,0]
    c2mask = data_t==[0,1]

    # c1はクラス1の学習データ
    c1 = data_x[c1mask]
    # c2はクラス2の学習データ
    c2 = data_x[c2mask]

    c1 = c1.reshape(int(c1.size/data_x.ndim), data_x.ndim)
    c2 = c2.reshape(int(c2.size/data_x.ndim), data_x.ndim)

    testOut1 = prob[:,0]
    testOut1 = testOut1.reshape(testX.shape)
    plt.contourf(testX,testY,testOut1,cmap='bwr')
    # 学習データを重ねて表示
    plt.plot(c1[:,0:1],c1[:,1:2],'o',mfc='red',mec='yellow')
    plt.plot(c2[:,0:1],c2[:,1:2],'s',mfc='blue',mec='yellow')
    plt.xlim([-3,3])
    plt.ylim([-3,3])


def main():
    ## 学習データの準備
    filename = 'classification_sample1.csv'
    train_x, train_y, validation_x, validation_y = common.loadData(filename)

    ## モデル
    model = LogisticRegression()

    ## 学習
    errs = model.fit(train_x, train_y)

    ## 学習過程の表示
    plot_learn(errs)

    ## 学習結果(予測と学習データ)の表示
    plot_eval(model, train_x, train_y)

    ## 検証データに対する評価
    accuracy = model.accuracy_score(validation_x, validation_y)
    print("accuracy=%f"%(accuracy))

    plt.show()


if __name__ == '__main__':
    main()
