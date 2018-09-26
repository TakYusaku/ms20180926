#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import common

class LeastMeanSquare:
    def __init__(self):
        self.W = 0

    def fit(self, data_x, data_t):
        # XとTを作る
        input_size = data_x.shape
        one = np.ones((input_size[0],1))
        Xtilde = np.c_[one, data_x] # ダミー値1を追加
        Ttilde = data_t

        # パラメータWを最小二乗法で求める
        self.W=np.linalg.solve(Xtilde.T.dot(Xtilde), Xtilde.T.dot(Ttilde))

        return self.accuracy_score(data_x, data_t)

    def predict(self, input):
        input_size = input.shape
        one = np.ones((input_size[0],1))
        Xtilde = np.c_[one, input] # ダミー値1を追加
        pred = self.W.T.dot(Xtilde.T).T

        ret = np.zeros(pred.shape)
        ret[pred[:,0]>pred[:,1]]=[1,0]
        ret[pred[:,0]<=pred[:,1]]=[0,1]
        return ret

    def accuracy_score(self, x, y):
        pred = self.predict(x)
        return accuracy_score(pred, y)

def main():
    ## 学習データの準備
    filename = 'classification_sample5.csv'
    train_x, train_y, validation_x, validation_y = common.loadData(filename)

    ## モデル
    model = LeastMeanSquare()

    ## 学習
    errs = model.fit(train_x, train_y)

    ## 学習過程の表示
    common.plot_learn(errs)

    ## 学習結果(予測と学習データ)の表示
    common.plot_eval(model, train_x, train_y)

    ## 検証データに対する評価
    accuracy = model.accuracy_score(validation_x, validation_y)
    print("accuracy=%f"%(accuracy))

    plt.show()


if __name__ == '__main__':
    main()
