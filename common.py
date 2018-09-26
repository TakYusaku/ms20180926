#! /usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def loadData(filename):
    data = np.loadtxt(filename, delimiter=',')
    train,test = train_test_split(data, random_state=1)
    return train[:,0:2], train[:,2:4], test[:,0:2], test[:,2:4]

def plot_learn(errs):
    print('error=%f'%(errs))

def plot_eval(model, data_x, data_t):
    ## 評価する
    # -3<x1<3, -3<x2<3の範囲でテストデータを生成
    testX,testY = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    testXin = testX.reshape(1,testX.size)
    testXin = testXin[0,:]
    testYin = testY.reshape(1,testY.size)
    testYin = testYin[0,:]

    test = np.array([testXin, testYin]).transpose()
    # 学習済みモデルに入力し，分類結果を予測
    predict = model.predict(test)

    # 描画
    c1mask = data_t==[1,0]
    c2mask = data_t==[0,1]

    # c1はクラス1の学習データ
    c1 = data_x[c1mask]
    # c2はクラス2の学習データ
    c2 = data_x[c2mask]

    c1 = c1.reshape(int(c1.size/data_x.ndim), data_x.ndim)
    c2 = c2.reshape(int(c2.size/data_x.ndim), data_x.ndim)

    # 予測を等高線表示に
    plt.figure(1)

    testOut1 = predict[:,0]
    testOut1 = testOut1.reshape(testX.shape)
    plt.contourf(testX,testY,testOut1,cmap='bwr')
    # 学習データを重ねて表示
    plt.plot(c1[:,0:1],c1[:,1:2],'o',mfc='red',mec='yellow')
    plt.plot(c2[:,0:1],c2[:,1:2],'s',mfc='blue',mec='yellow')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    
