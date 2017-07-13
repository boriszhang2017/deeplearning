#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV


def plot_decision_boundary(pred_func, x, y, nm_hid=0):
    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    h = 0.01
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h)
                           , np.arange(x2_min, x2_max, h))
    z = pred_func(np.c_[xx1.ravel(), xx2.ravel()])
    z = z.reshape(xx1.shape)

    plt.contour(xx1, xx2, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=60, cmap=plt.cm.Spectral)
    plt.title("Logistic Regression")
    # plt.title("Decision Boundary for hidden layer size " + str(nm_hid))
    plt.show()


def build_model_ann(x, nm_hid, nm_iter=20000, print_loss=False):
    """
    参数说明：
        nm_hid: 隐层节点数
        nm_iter: 梯度下降迭代次数
        print_loss: 为真时每1000次打印结果
    """

    # parameters for gradient descent
    epsilon = 0.01  # 学习率
    reg_lambda = 0.01  # 正则化参数

    nm_samples = len(x)
    nm_input_dim = 2   # 2 input dimensions
    nm_output_dim = 2  # 2 output classes
    # 随机初始化权重
    np.random.seed(0)
    # np.random.randn: 从标准正态分布中返回一个或多个样本值。
    W1 = np.random.randn(nm_input_dim, nm_hid) / np.sqrt(nm_input_dim)
    b1 = np.zeros((1, nm_hid))
    W2 = np.random.randn(nm_hid, nm_output_dim) / np.sqrt(nm_hid)
    b2 = np.zeros((1, nm_output_dim))

    model = {}  # ann model
    for i in xrange(0, nm_iter):    # 梯度下降
        # 前向运算计算loss
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs
        delta3[range(nm_samples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 加上正则化项，防止过拟合
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 梯度下降更新参数
        W1 += -epsilon * dW1
        W2 += -epsilon * dW2
        b1 += -epsilon * db1
        b2 += -epsilon * db2
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if  print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" \
                  % (i, calc_loss(model, reg_lambda, nm_samples))
    return model

# 计算softmax交叉熵损失
def calc_loss(model, reg_lambda, nm_samples):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 前向运算计算loss
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)     # softmax
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_logprobs = -np.log(probs[range(nm_samples), y])
    data_loss = np.sum(corect_logprobs)
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    data_loss = 1.0 / nm_samples * data_loss
    return data_loss


def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 前向运算计算loss
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)  # softmax
    # 计算概率输出最大概率对应的类别
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


if __name__ == "__main__":
    x, y = make_moons(200, noise=0.1, random_state=0)
    # plt.scatter(x[:,0], x[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    # ******************************************************
    # Example 1: logistic model
    # ******************************************************
    clf = LogisticRegressionCV()
    clf.fit(x, y)
    plot_decision_boundary(lambda x: clf.predict(x), x, y)

    # ******************************************************
    # Example 2: ANN
    # ******************************************************
    # nm_hid = 4
    # model = build_model_ann(x, nm_hid, print_loss=True)
    # # print model
    # plot_decision_boundary(lambda t: predict(model, t), x, y, nm_hid)




