import math
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True


class GMM(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def gauss_density(self, x, mean, sigma):
        """caculate gauss density"""

        return 1.0 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    def multi_gauss_density(self, x, mean, sigma):
        """caculate multi-demensional gauss density"""

        coef = 1.0 / math.sqrt(pow(2 * math.pi, len(x)) * np.linalg.det(sigma))
        exponent = math.exp((-(x - mean) * sigma.I * (x - mean).reshape(len(mean), 1) / 2.0)[0, 0])
        return coef * exponent

    def generate_data(self, alpha, means, sigma, data_num):
        """generate one-demensional data"""

        initial_data = []
        for i in range(data_num):
            prob_sum = 0
            rand = np.random.random()
            for index in range(len(alpha)):
                prob_sum += alpha[index]
                if rand < prob_sum:
                    initial_data.append(np.random.normal(means[index], sigma[index]))
                    break
        return initial_data

    def generate_multi_data(self, alpha, means, sigma, data_num):
        """generate multi-demensional data"""

        initial_data = []
        for i in range(data_num):
            prob_sum = 0
            rand = np.random.random()
            for index in range(len(alpha)):
                prob_sum += alpha[index]
                if rand < prob_sum:
                    initial_data.append(np.random.multivariate_normal(means[index], sigma[index]))
                    break
        return initial_data

    def EM(self, initial_data, alpha, means, sigma, max_iter):
        """EM algorithm for one-demensional GMM"""

        gauss_num, data_num = len(alpha), len(initial_data)
        iter_cnts, gama_arr = 0, np.zeros((gauss_num, data_num), dtype=np.float64)
        while iter_cnts < max_iter:
            old_alpha, old_means, old_sigma = alpha.copy(), means.copy(), sigma.copy()

            # caculate model responsivity
            for k in range(gauss_num):
                for j in range(data_num):
                    gauss_densities = [self.gauss_density(initial_data[j], means[t], sigma[t]) for t in range(gauss_num)]
                    gama_arr[k][j] = alpha[k] * self.gauss_density(initial_data[j], means[k], sigma[k]) / alpha.dot(gauss_densities)

            # update arguments
            for k in range(gauss_num):
                alpha[k] = sum(gama_arr[k]) / data_num
                means[k] = gama_arr[k].dot(initial_data) / sum(gama_arr[k])
                sigma[k] = math.sqrt(gama_arr[k].dot((initial_data - means[k]) ** 2) / sum(gama_arr[k]))

            iter_cnts += 1
            if DEBUG: print(alpha, means, sigma)
            if sum((old_alpha - alpha) ** 2 + (old_means - means) ** 2 + (old_sigma - sigma) ** 2) < self.epsilon: break
        return [alpha, means, sigma], iter_cnts

    def multi_EM(self, initial_data, alpha, means, sigma, max_iter):
        """EM algorithm for multi-demensional GMM"""

        dim, gauss_num, data_num = len(means[0]), len(alpha), len(initial_data)
        iter_cnts, gama_arr = 0, np.zeros((gauss_num, data_num), dtype=np.float64)
        while iter_cnts < max_iter:
            old_alpha, old_means, old_sigma = alpha.copy(), means.copy(), sigma.copy()

            # caculate model responsivity
            for k in range(gauss_num):
                for j in range(data_num):
                    gauss_densities = [self.multi_gauss_density(initial_data[j], means[t], sigma[t]) for t in range(gauss_num)]
                    gama_arr[k][j] = alpha[k] * self.multi_gauss_density(initial_data[j], means[k], sigma[k]) / alpha.dot(gauss_densities)

            # update arguments
            for k in range(gauss_num):
                alpha[k] = sum(gama_arr[k]) / data_num
                means[k] = gama_arr[k].dot(initial_data) / sum(gama_arr[k])
                tmp_value = np.array([data.reshape(dim, 1).dot(data.reshape(1, dim)) for data in initial_data - means[k]])
                sigma[k] = np.matrix(sum([gama_arr[k][i] * tmp_value[i] for i in range(data_num)]) / sum(gama_arr[k]))

            iter_cnts += 1
            if DEBUG: print(alpha, [list(item) for item in means])
            if sum((old_alpha - alpha) ** 2) + sum(
                    [sum((old_means[i] - means[i]) ** 2) for i in range(gauss_num)]) < self.epsilon: break
        return [alpha, means, sigma], iter_cnts

    def cluster(self, initial_data, alpha, means, sigma):
        """mark data by the trained arguments"""

        classified, gauss_num = [], len(alpha)
        for item in initial_data:
            probs = [alpha[k] * self.multi_gauss_density(item, means[k], sigma[k]) for k in range(gauss_num)]
            classified.append(probs.index(max(probs)))
        return classified

    def gmm_one_dimension_test(self):
        """test for one-dimensional data"""

        lab = GMM(epsilon=1e-5)
        data_num, max_iter = 500, 100

        alpha = np.array([0.2, 0.5, 0.3])
        means = np.array([20.0, 10.0, 5.0])
        sigma = np.array([2.0, 1.0, 0.2])

        initial_alpha = np.array([1 / len(alpha) for _ in range(len(alpha))])
        initial_means = np.array([15.0, 5.0, 1.0])
        initial_sigma = np.array([1.0, 1.0, 1.0])

        initial_data = lab.generate_data(alpha, means, sigma, data_num)
        parameters, iter_cnts = lab.EM(initial_data, initial_alpha, initial_means, initial_sigma, max_iter)
        return parameters, iter_cnts

    def gmm_two_dimension_test(self):
        """test for two-dimensional data"""

        lab = GMM(epsilon=1e-5)
        data_num, max_iter = 500, 100

        alpha = np.array([0.20, 0.30, 0.50])

        means = [np.array([5.0, 20.0]),
                 np.array([15.0, 30.0]),
                 np.array([30.0, 20.0])]

        sigma = [np.matrix([[9.0, 0.0], [0.0, 4.0]]),
                 np.matrix([[4.0, 0.0], [0.0, 7.0]]),
                 np.matrix([[6.0, 0.0], [0.0, 4.0]])]

        initial_alpha = np.array([1 / len(alpha) for _ in range(len(alpha))])
        initial_means = [np.array([10.0, 10.0]),
                         np.array([20.0, 20.0]),
                         np.array([30.0, 30.0])]
        initial_sigma = [np.matrix([[1.0, 0.0], [0.0, 1.0]]),
                         np.matrix([[1.0, 0.0], [0.0, 1.0]]),
                         np.matrix([[1.0, 0.0], [0.0, 1.0]])]

        initial_data = lab.generate_multi_data(alpha, means, sigma, data_num)
        parameters, iter_cnts = lab.multi_EM(initial_data, initial_alpha, initial_means, initial_sigma, max_iter)
        classified = lab.cluster(initial_data, *parameters)

        xpts = [x for x, y in initial_data]
        ypts = [y for x, y in initial_data]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(xpts, ypts, c='b')
        plt.subplot(1, 2, 2)
        plt.scatter(xpts, ypts, c=classified)
        plt.savefig('EM_for_GMM.jpg')
        if DEBUG: plt.show()
        return parameters, iter_cnts

    def gmm_uci_dataset_test(self):
        """test for uci dataset"""

        lab = GMM(epsilon=1e-5)
        data_num, max_iter = 500, 100
        flowers = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

        initial_data, categories = [], []
        with open('data_for_gmm.txt') as fr:
            for line in fr:
                item = line.strip().split(',')
                initial_data.append(np.array(item[:4], dtype=np.float))
                categories.append(flowers[item[-1]])

        initial_alpha = np.array([1 / 3 for _ in range(3)])
        initial_means = [np.array([5.0, 3.2, 1.5, 0.2]),
                         np.array([6.0, 2.5, 4.5, 1.2]),
                         np.array([7.0, 3.0, 5.5, 2.0])]
        initial_sigma = [np.matrix(np.identity(4)) for _ in range(3)]

        parameters, iter_cnts = lab.multi_EM(initial_data, initial_alpha, initial_means, initial_sigma, max_iter)
        classified = lab.cluster(initial_data, *parameters)
        correct_rate = sum([1 if item[0] == item[1] else 0 for item in zip(categories, classified)]) / len(categories)
        print("\ncorrect_rate =", correct_rate)
        return parameters, iter_cnts

    def print_result(self, parameters, iter_cnts):
        """print parameters and iter_cnts for gmm_test"""

        print("\n迭代次数:", iter_cnts)

        print("\n估计系数:")
        for item in parameters[0]: print(item)

        print("\n估计均值:")
        for item in parameters[1]: print(item.tolist())

        print("\n估计方差:")
        for item in parameters[2]: print(item.tolist())


if __name__ == '__main__':
    lab = GMM(epsilon=1e-5)
    data_num, max_iter = 500, 100

    print('result for gmm_one_dimension_test:')
    lab.print_result(*lab.gmm_one_dimension_test())

    print('\n\n\nresult for gmm_two_dimension_test:')
    lab.print_result(*lab.gmm_two_dimension_test())

    print('\n\n\nresult for gmm_uci_dataset_test:')
    lab.print_result(*lab.gmm_uci_dataset_test())
