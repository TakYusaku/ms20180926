#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import perceptron

import common

class LeastMeanSquare:
    def __init__(self):
        self.net = perceptron.Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)

    def fit(self, data_x, data_t):
        self.net.fit(data_x, data_t[:,0])

        error = np.mean(np.square(data_t-self.predict(data_x)))/2
        return error

    def predict(self, input):
        pred = self.net.predict(input)
        ret = np.c_[pred,1-pred]
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
