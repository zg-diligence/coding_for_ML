"""
  Copyright(c) 2017 Gang Zhang
  All rights reserved.
  Author:Gang Zhang
  Date:2017.9.30
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']

"""
    step_size = reverseParaCurve(a, b, reverseParaCurve(a, b, c, gradient), gradient) # 逆抛物线法
    # step_size = (gradient.T * gradient / (gradient.T * hesse * gradient)).A[0][0]   # 线性搜索
"""

DEBUG = True

class Curvefitting:
    def __init__(self, M, N, X, Y):
        unit_matrix = [[0 for _ in range(M + 1)] for _ in range(M + 1)]
        for i in range(M + 1): unit_matrix[i][i] = 1

        self.M = M  # 幂指数
        self.N = N  # 样本量

        self.steps = [0.05, 0.3, 0.5, 1]                        # 候选步长
        self.unit_matrix = np.matrix(unit_matrix)               # 单位矩阵
        self.sumSquareVector = lambda vec: np.dot(vec.T, vec)   # 向量模方

        self.X = X  # 样本点纵坐标
        self.Y = Y  # 样本点纵坐标
        self.W = np.matrix([random.randint(15000, 20000) for _ in range(M + 1)]).T      # 代价方程待求系数
        self.A = np.matrix([[math.pow(x, n) for n in range(M + 1)] for x in self.X])    # 代价方程系数矩阵
        self.Q = self.A.T * self.A                                                      # 代价方程正定矩阵

    def calcCost(self, alpha, W=None):
        """计算代价函数"""
        if W is None: W = self.W
        diff = self.A * W - self.Y
        return (diff.T * diff + alpha * W.T * W) / 2
    
    def calcGradient(self, alpha, W=None):
        """计算梯度"""
        if W is None: W = self.W
        return self.A.T * (self.A * W - self.Y) + alpha * W

    def leastSquareMethod(self):
        """最小二乘法"""
        args = (self.A.T * self.A).I * self.A.T * self.Y
        return [x[0] for x in args.A]

    def batchGradientDescent(self, alpha=0.0, eplison=1e-10, max_iter=1000000):
        """批量梯度下降法"""
        W, iter_cnts = self.W, 0
        cost = self.calcCost(alpha).A[0][0] # 根据初始参数值计算代价
        while True:
            iter_cnts += 1
            if iter_cnts > max_iter: break  # 超出迭代次数

            gradient = self.calcGradient(alpha, W)
            if self.sumSquareVector(gradient) < eplison: break  # 梯度二范数平方值达到迭代精度
            new_weights = [W - step_size * gradient for step_size in self.steps]            # 计算新参数值
            new_costs = [self.calcCost(alpha, weight).A[0][0] for weight in new_weights]    # 计算新的代价
            index = new_costs.index(min(new_costs))
            new_W, new_cost = new_weights[index], new_costs[index]
            if cost - new_cost < eplison: break # 代价函数值达到迭代精度
            W, cost = new_W, new_cost           # 更新参数值、代价函数值
            if DEBUG and iter_cnts % 10000 == 0: print(self.steps[index], cost)
        return iter_cnts, [x[0] for x in W.A]

    def conjugateGradientDescent(self, alpha=0.0, eplison=1e-10, max_iter=1000000):
        """共轭梯度下降法"""
        gradient = self.calcGradient(alpha)
        iter_cnts, direction, W = 0, -gradient, self.W
        while True:
            iter_cnts += 1
            if iter_cnts > max_iter: break  # 超出迭代次数

            if self.sumSquareVector(gradient) < eplison: break  # 梯度二范数平方值达到迭代精度
            step_size = (-gradient.T * direction / (direction.T * self.Q * direction)).A[0][0]     # 计算步长
            new_W = W + step_size * direction           # 计算新的参数值
            gradient = self.calcGradient(alpha, new_W)  # 计算新的梯度值
            beta = (gradient.T * self.Q * direction / (direction.T * self.Q * direction)).A[0][0]  # 计算方向系数beta
            direction, W = -gradient + beta * direction, new_W  # 更新方向和参数值
        return iter_cnts, [x[0] for x in W.A]

    def reverseParaCurve(self, a, b, c, W, gradient, alpha):
        """计算过a, b, c三点抛物线最低点的位置"""
        weights = [W - step_size * gradient for step_size in (a, b, c)]
        costs = [self.calcCost(alpha, weight).A[0][0] for weight in weights]
        numerator = ((b-a)**2)*(costs[1] - costs[2]) - ((b - c)**2)*(costs[1] - costs[0])
        denominator = (b-a)*(costs[1] - costs[2]) - (b - c)*(costs[1] - costs[0])
        return b - 0.5 * numerator / denominator

    def showFittingResult(self, args, folder):
        """展示拟合曲线"""
        sin_X = np.linspace(0.1, 1, 200)
        sin_Y = [math.sin(2 * math.pi * x) for x in sin_X]
        matrix_sin_X = np.matrix([[math.pow(x, n) for n in range(self.M + 1)] for x in sin_X])
        fitting_Y = [np.dot(row, args) for row in matrix_sin_X.A]

        plt.figure()
        for x, y in list(zip(self.X, self.Y)): plt.plot(x, y, 'yo')
        plt.title("M = " + str(self.M) + ", N = " + str(self.N))
        plt.plot(sin_X, sin_Y)
        plt.plot(sin_X, fitting_Y)
        plt.savefig(folder + "/" + str(self.M))

    def standardDeviation(self, args, X, Y):
        """计算训练误差和测试误差"""
        sin_X, sin_Y = X, Y
        matrix_sin_X = np.matrix([[math.pow(x, n) for n in range(self.M + 1)] for x in sin_X])

        fitting_Y = [np.dot(row, args) for row in self.A.A]
        train_deviation = math.sqrt(sum([(a-b)**2 for a,b in zip(self.Y, fitting_Y)])/10)
        fitting_Y = [np.dot(row, args) for row in matrix_sin_X.A]
        test_deviation = math.sqrt(sum([(a-b)**2 for a,b in zip(sin_Y, fitting_Y)])/len(sin_Y))
        return train_deviation, test_deviation


def seekBestAlpha(alpha_set, X, Y, func_pos=1):
    """遍历寻找最佳正则项系数"""
    sin_X = np.linspace(0.1, 1, 100)
    sin_Y = [math.sin(2 * math.pi * x) for x in sin_X]
    matrix_sin_X = np.matrix([[math.pow(x, n) for n in range(10)] for x in sin_X])

    ret = []
    for alpha in alpha_set:
        fitter = Curvefitting(9, 10, X, Y)
        if func_pos == 1:
            args = fitter.conjugateGradientDescent(alpha=alpha)[1]
        else:
            args = fitter.batchGradientDescent(alpha=alpha)[1]
        fitting_Y = [np.dot(row, args) for row in matrix_sin_X.A]
        cost = sum([(a-b)**2 for a,b in zip(sin_Y, fitting_Y)])
        ret.append((alpha, cost))
    return ret


if __name__ == '__main__':
    # 训练集
    N = 10
    X = [x / N for x in range(1, N + 1)]
    Y = np.matrix([math.sin(2 * math.pi * x) + np.random.normal(0, 0.2) for x in X]).T

    # 测试集
    sin_X = np.linspace(0.1, 1, 200)
    sin_Y = [math.sin(2 * math.pi * x) for x in sin_X]

    # 训练集大小对模型的影响
    if not os.path.exists('add_points'):
        os.mkdir('add_points')
    with open('add_points/args.txt', 'w') as f:
        for N in [10, 20, 50, 100, 200]:
            X = [x / N for x in range(1, N + 1)]
            Y = np.matrix([math.sin(2 * math.pi * x) + np.random.normal(0, 0.2) for x in X]).T
            fitter = Curvefitting(9, N, X, Y)
            arguments = fitter.leastSquareMethod()
            f.write("N = " + str(N) + ":" + str(arguments) + "\n")
            fitter.showFittingResult(arguments, 'add_points')

    # 计算最佳惩罚项系数
    alpha_set = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    X_ticks = [str(x) for x in alpha_set]
    result = seekBestAlpha(alpha_set, X, Y)
    plt_y = [b for (a, b) in result]
    plt.figure()
    plt.plot(list(range(len(plt_y))), plt_y)
    plt.xticks(list(range(len(plt_y))), X_ticks)
    plt.xlabel("lambda")
    plt.ylabel("测试误差")
    plt.savefig("penaltys")

    # 最小二乘法
    if not os.path.exists('least_square_method'):
        os.mkdir('least_square_method')
    with open('least_square_method/args.txt', 'w') as f:
        train_devs, test_devs = [], []
        for M in range(0, 10):
            print("M =", M)
            fitter = Curvefitting(M, N, X, Y)
            arguments = fitter.leastSquareMethod()
            f.write("M = " + str(M) + ":" + str(arguments) + "\n")
            fitter.showFittingResult(arguments, 'least_square_method')
            train_dev, test_dev = fitter.standardDeviation(arguments, sin_X, sin_Y)
            train_devs.append(train_dev)
            test_devs.append(test_dev)
        plt.figure()
        plt.plot(list(range(10)), train_devs),
        plt.plot(list(range(10)), test_devs)
        plt.savefig("deviations_1")

    # 批量梯度下降
    if not os.path.exists('batch_gradient_descent'):
        os.mkdir('batch_gradient_descent')
    with open('batch_gradient_descent/args.txt', 'w') as f:
        train_devs, test_devs = [], []
        for M in range(0, 10):
            print("M =", M)
            fitter = Curvefitting(M, N, X, Y)
            alpha = 1e-5 if M == 9 else 0
            arguments = fitter.batchGradientDescent(alpha=alpha)[1]
            f.write("M = " + str(M) + ":" + str(arguments) + "\n")
            fitter.showFittingResult(arguments, 'batch_gradient_descent')
            train_dev, test_dev = fitter.standardDeviation(arguments, sin_X, sin_Y)
            train_devs.append(train_dev)
            test_devs.append(test_dev)
        plt.figure()
        plt.plot(list(range(10)), train_devs),
        plt.plot(list(range(10)), test_devs)
        plt.savefig("deviations_2")

    # 共轭梯度法
    if not os.path.exists('conjugate_gradient_descent'):
        os.mkdir('conjugate_gradient_descent')
    with open('conjugate_gradient_descent/args.txt', 'w') as f:
        train_devs, test_devs = [], []
        for M in range(0, 10):
            print("M =", M)
            fitter = Curvefitting(M, N, X, Y)
            alpha = 1e-5 if M == 9 else 0
            arguments = fitter.conjugateGradientDescent(alpha=alpha)[1]
            f.write("M = " + str(M) + ":" + str(arguments) + "\n")
            fitter.showFittingResult(arguments, 'conjugate_gradient_descent')
            train_dev, test_dev = fitter.standardDeviation(arguments, sin_X, sin_Y)
            train_devs.append(train_dev)
            test_devs.append(test_dev)
        plt.figure()
        plt.plot(list(range(10)), train_devs),
        plt.plot(list(range(10)), test_devs)
        plt.savefig("deviations_3")
