import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris
import tensorflow as tf

import common
import iris_plot
import ex3_LR

class LR(ex3_LR.LogisticRegression):
    def __init__(self):
        # set up LogisticRegression
        self.dim_input = 4
        self.dim_output = 3

        # imput x and training data t
        self.x = tf.placeholder(tf.float32, [None, self.dim_input], name="x-input")
        self.t = tf.placeholder(tf.float32, [None, self.dim_output], name="t-teacher")

        # output y
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

    def predict(self, input):
        pred = self.probability(input)
        target =np.argmax(pred,axis=1)
        pred = np.eye(3)[target].astype("int")
        return pred

def plot_learn(errs):
    ## 学習過程を表示
    max_epoch = len(errs)
    plt.figure(2)
    plt.plot(np.arange(0,max_epoch), errs, 'k')
    plt.xlabel('epoch')
    plt.ylabel('error(cross entropy)')


def _main():
    iris = load_iris()
    train_x, validation_x, train_y, validation_y = train_test_split(iris['data'],iris['target'])
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y) # 1 of K 表記に変換 (元は 0,1,2 のラベル)
    validation_y = lb.fit_transform(validation_y)
    res = []

    for i in range(1):
        ## モデル
        model = LR()

        ## 学習
        errs = model.fit(train_x, train_y)

        ## 検証データに対する評価
        accuracy = model.accuracy_score(validation_x, validation_y)
        res.append(accuracy)
        print("accuracy=%f"%(accuracy))

        if i == 0:
            print(np.average(np.array(res)))
            ## 学習過程の表示
            plot_learn(errs)

            ## 学習結果(予測と学習データ)の表示
            rst = iris_plot.plot_eval(model, train_x, train_y, validation_x, validation_y)
            plt.show()


if __name__ == '__main__':
    _main()
