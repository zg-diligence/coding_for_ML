"""
  Copyright(c) 2017 Gang Zhang
  All rights reserved.
  Author:Gang Zhang
  Date:2017.10.27
"""

import math
import codecs
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

plt.rcParams['font.sans-serif']=['SimHei']

DEBUG = True
CURVE = False

class LR:
    def __init__(self, theta, epsilon, nlambda):
        self.theta = theta
        self.epsilon = epsilon
        self.nlambda = nlambda

    @staticmethod
    def sigmoid(x):
        return 1.0/(1+np.exp(-x))

    @staticmethod
    def calc_cost(x, y, theta, nlambda):
        """计算代价函数值"""
        pix = LR.sigmoid(np.dot(theta, x.T))
        result = [y_i*np.log(pix_i) + (1-y_i)*np.log(1-pix_i) for y_i, pix_i in zip(y, pix)]
        return (-np.sum(result, 0) + 0.5*nlambda*np.dot(theta, theta))/len(y)

    @staticmethod
    def calc_gradient(x, y, theta, nlambda):
        """计算梯度"""
        hx = LR.sigmoid(np.dot(theta, x.T))
        return (np.sum([[item[0] * v for v in item[1]] for item in zip(hx - y, x)], 0)  + nlambda*theta)/ len(x)

    @staticmethod
    def calc_hessian(x, theta, nlambda):
        """计算黑塞矩阵"""
        hx = LR.sigmoid(np.dot(theta, x.T))
        A = hx*(1-hx)*np.eye(len(x))
        return (nlambda*np.eye(len(theta)) + np.mat(x.T) * A * np.mat(x))/ len(x)

    @staticmethod
    def calc_best_alpha(x, y, theta, nlambda, dk):
        """Armijo搜索步长"""
        m, mk = 0, 0
        rho, sigma = 0.5, 0.4
        while m < 20:
            value1 = LR.calc_cost(x + rho**m*dk, y, theta, nlambda)
            value2 = LR.calc_cost(x, y, theta, nlambda) + sigma*rho**m*np.dot(dk, -dk)
            if value1 < value2:
                mk = m; break
            m += 1
        return rho**mk

    def gradient_descent_1(self, x, y, alpha, max_iter):
        """定步长梯度下降法"""
        theta = self.theta
        epsilon = self.epsilon

        iter_cnts = 1
        row, _ = np.shape(x)
        while iter_cnts <= max_iter:
            gradient = LR.calc_gradient(x, y, theta, self.nlambda)
            if DEBUG: print(gradient)
            if np.dot(gradient, gradient) < epsilon: break
            theta -= alpha * gradient
            iter_cnts += 1

        if not CURVE: LR.display_line(x, y, theta, iter_cnts, 'result_1.jpg')
        if CURVE: LR.display_curve(x, y, list(theta), iter_cnts, 'result_1.jpg')
        return list(theta)

    def gradient_descent_2(self, x, y, max_iter):
        """Armijo搜索步长 梯度下降法"""
        theta = self.theta
        epsilon = self.epsilon

        iter_cnts = 1
        row, _ = np.shape(x)
        while iter_cnts <= max_iter:
            gradient = LR.calc_gradient(x, y, theta, self.nlambda)
            if DEBUG: print(gradient)
            if np.dot(gradient, gradient) < epsilon:break
            best_alpha = LR.calc_best_alpha(x, y, theta, self.nlambda, -gradient)
            theta -= best_alpha * gradient
            iter_cnts += 1

        if not CURVE: LR.display_line(x, y, theta, iter_cnts, 'result_2.jpg')
        if CURVE: LR.display_curve(x, y, list(theta), iter_cnts, 'result_2.jpg')
        return theta

    def newton(self, x, y, max_iter):
        """牛顿法"""
        theta = self.theta
        epsilon = self.epsilon

        iter_cnts = 1
        while iter_cnts <= max_iter:
            gradient = LR.calc_gradient(x, y, theta, self.nlambda)     # 计算梯度
            if DEBUG: print(gradient)
            if np.dot(gradient, gradient) < epsilon: break
            hessian = LR.calc_hessian(x, theta, self.nlambda)          # 计算黑塞矩阵
            dk = -np.linalg.solve(hessian, gradient)                   # 确定搜索方向
            best_alpha = LR.calc_best_alpha(x, y, theta, self.nlambda, dk)  # Armijo搜索步长
            theta += best_alpha * dk                                   # 更新参数值
            iter_cnts += 1

        if not CURVE: LR.display_line(x, y, theta, iter_cnts, 'result_3.jpg')
        if CURVE: LR.display_curve(x, y, list(theta), iter_cnts, 'result_3.jpg')
        return theta

    def DFP(self, x, y, max_iter):
        """DFP算法"""
        theta = self.theta
        epsilon = self.epsilon

        iter_cnts = 1
        row, col = np.shape(x)
        Hk = np.eye(col)
        while iter_cnts <= max_iter:
            gradient = LR.calc_gradient(x, y, theta, self.nlambda)          # 计算梯度
            if DEBUG: print(gradient)
            if np.dot(gradient, gradient) < epsilon:break
            dk = -np.dot(Hk, gradient)                                      # 确定搜索方向
            best_alpha = LR.calc_best_alpha(x, y, theta, self.nlambda, dk)  # Armijo搜索步长

            new_theta = theta + best_alpha*dk
            new_gradient = LR.calc_gradient(x, y, new_theta, self.nlambda)
            sk = new_theta - theta
            yk = new_gradient - gradient

            # 校正近似黑塞矩阵
            if np.dot(sk, yk) > 0:
                Hy = np.dot(Hk, yk)
                yHy = np.dot(yk, Hy)
                Hk -= (Hy.reshape(col, 1)*Hy/yHy - sk.reshape((col, 1))*sk/np.dot(sk, yk))
            iter_cnts += 1
            theta = new_theta

        if not CURVE: LR.display_line(x, y, theta, iter_cnts, 'result_4.jpg')
        if CURVE: LR.display_curve(x, y, list(theta), iter_cnts, 'result_4.jpg')
        return theta

    def BFGS(self, x, y, max_iter):
        """BFGS算法"""
        theta = self.theta
        epsilon = self.epsilon

        iter_cnts = 1
        row, col = np.shape(x)
        Bk = np.eye(col)
        while iter_cnts <= max_iter:
            gradient = LR.calc_gradient(x, y, theta, self.nlambda)          # 计算梯度值
            if DEBUG: print(gradient)
            if np.dot(gradient, gradient) < epsilon: break
            dk = -np.linalg.solve(Bk, gradient)                             # 确定搜索方向
            best_alpha = LR.calc_best_alpha(x, y, theta, self.nlambda, dk)  # ARrmijo搜索步长

            new_theta = theta + best_alpha * dk
            new_gradient = LR.calc_gradient(x, y, new_theta, self.nlambda)
            sk = new_theta - theta
            yk = new_gradient - gradient

            # 校正近似黑塞矩阵
            if np.dot(sk, yk) > 0:
                Bs = np.dot(Bk, sk)
                sBs = np.dot(sk, Bs)
                Bk -= (Bs.reshape(col, 1) * Bs / sBs - yk.reshape((col, 1)) * yk / np.dot(yk, sk))
            iter_cnts += 1
            theta = new_theta

        if not CURVE: LR.display_line(x, y, theta, iter_cnts, 'result_5.jpg')
        if CURVE: LR.display_curve(x, y, theta, iter_cnts, 'result_5.jpg')
        return theta

    @staticmethod
    def display_line(x, y, theta, iter_cnts, filename):
        """可视化分类结果, 线性决策面"""
        plt.figure()
        for i in range(len(x)):
            plt.plot(x[i, 0], x[i, 1], ('bo' if y[i] else 'ro'))

        xpoints = np.linspace(-1.5, 2.5)
        ypoints = (theta[0] * xpoints + theta[2]) / (-theta[1])
        plt.plot(xpoints, ypoints, 'g-', lw=2)

        plt.xlim([-3, 3])
        plt.ylim([-1.5, 2.5])
        plt.title('Iteration = ' + str(iter_cnts))
        plt.savefig(filename, dpi=200, bbox_inches='tight')

    @staticmethod
    def display_curve(X, Y, theta, iter_cnts, filename):
        """可视化分类结果, 非线性决策面"""
        plt.figure(figsize=(6, 6))
        for i in range(len(X)):
            plt.plot(X[i, 0], X[i, 1], ('bo' if Y[i] else 'ro'))

        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)
        f = theta[0]*x + theta[1]*y + theta[2]*x**2 + theta[3]*y**2 + theta[4]*x*y + theta[5]
        data = plt.contour(x, y, f, levels=[0, ]).collections[0].get_paths()[0].vertices
        plt.plot(data[:, 0], data[:, 1], 'r')

        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.title('Iteration = ' + str(iter_cnts))
        plt.savefig(filename, dpi=200, bbox_inches='tight')

    @staticmethod
    def generate_data():
        """手动生成数据, 非线性决策面"""
        points, label = [], []

        # 产生分类1的数据
        for _ in range(200):
            r = random.uniform(0, 1.8)
            angle = random.uniform(0, 2)
            x = r * math.sin(angle * math.pi)
            y = r * math.cos(angle * math.pi)
            points.append((x, y, x**2, y**2, x*y, 1))
            label.append(0)

        # 产生分类2的数据
        for _ in range(500):
            r = random.uniform(2.2, 4)
            angle = random.uniform(0, 2)
            x = r * math.sin(angle * math.pi)
            y = r * math.cos(angle * math.pi)
            points.append((x, y, x**2, y**2, x*y, 1))
            label.append(1)

        # 产生噪声数据
        # for i in range(10):
        #     r = random.uniform(1.5, 2.5)
        #     angle = random.uniform(0, 2)
        #     x = r * math.sin(angle * math.pi)
        #     y = r * math.cos(angle * math.pi)
        #     points.append((x, y, x**2, y**2, x*y, 1))
        #     label.append(i%2)

        return points, label

    @staticmethod
    def read_real_data(filename):
        """读取真实数据集"""
        with codecs.open(filename, 'r', encoding='utf8') as fr:
            data = [line.strip() for line in fr]
            features, labels = [], [[] for _ in range(3)]
            for item in data:
                parts = item.split(',')
                # features.append([float(parts[i]) for i in range(4)] + [1.0, ])
                features.append([float(parts[i]) for i in range(4)] + [float(parts[i])**2 for i in range(4)] + [1.0, ])
                labels[0].append(1 if parts[4]=='Iris-setosa' else 0)
                labels[1].append(1 if parts[4]=='Iris-versicolor' else 0)
                labels[2].append(1 if parts[4]=='Iris-virginica' else 0)
            return features, labels

    @staticmethod
    def classify(x, theta):
        """对测试集进行分类"""
        return [1 if np.dot(item, theta) >=0 else 0 for item in x]

    @staticmethod
    def calc_accuracy(predict, origin):
        """计算模型在测试集上的准确率"""
        return np.sum([1 if pred==orig else 0 for pred,orig in zip(predict, origin)])/len(origin)


if __name__ == '__main__':
    nlambda = 0  # 惩罚项系数
    max_iter = 1000                     # 最大迭代次数

    # 1.真实数据集测试
    # theta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0]
    # x, ylabels = LR.read_real_data('train.txt')
    # operation = LR(np.array(theta), 1e-5, nlambda)
    # for i in range(3):
    #     test_data, test_label = [], []
    #     train_data, train_label = [], []
    #     for j in range(len(ylabels[0])):
    #         if j%3 == 0:  # 训练集100、测试集50
    #             test_data.append(x[j])
    #             test_label.append(ylabels[i][j])
    #         else:
    #             train_data.append(x[j])
    #             train_label.append(ylabels[i][j])
    #
    #     train_data = np.array(train_data)
    #     train_label = np.array(train_label)
    #
    #     theta = operation.gradient_descent_1(train_data, train_label, 0.5, max_iter)
    #     result = LR.classify(test_data, theta)
    #     print(theta, end='\t')
    #     print('Accuracy of the ' + str(i) + 'th category:', LR.calc_accuracy(result, test_label))

    # 2.手工生成数据 -- 非线性决策面
    # theta = [1.0, 1.0, 1.0, 1.0, 1.0, 0]
    # x, y = LR.generate_data()
    # x = np.array(x)
    # y =np.array(y)

    # # 3.skleran数据 -- 线性决策面
    theta = [1.0, 1.0, 0]
    x, y = make_moons(500, noise=0.25)
    x = [list(row) for row in x]
    for row in x: row.append(1)
    x = np.array(x)

    operation1 = LR(np.array(theta), 1e-5, nlambda)
    print(operation1.gradient_descent_1(x, y, 0.5, max_iter))

    operation2 = LR(np.array(theta), 1e-5, nlambda)
    print(operation2.gradient_descent_2(x, y, max_iter))

    operation3 = LR(np.array(theta), 1e-5, nlambda)
    print(operation3.newton(x, y, max_iter))

    operation4 = LR(np.array(theta), 1e-5, nlambda)
    print(operation4.DFP(x, y, max_iter))

    operation5 = LR(np.array(theta), 1e-5, 0.5)
    print(operation5.BFGS(x, y, max_iter))
