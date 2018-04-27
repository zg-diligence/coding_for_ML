"""
  Copyright(c) 2017 Gang Zhang
  All rights reserved.
  Author:Gang Zhang
  Date:2017.11.24
"""

import os
import cv2
import math
import heapq
import numpy as np
from PIL import Image
from code_of_ML.EM_GMM.GMM import GMM
from matplotlib import image
from matplotlib import pyplot as plt

DEBUG = True
img_shape = (100, 100)


class PCA(object):
    def __init__(self):
        pass

    def gauss_density(self, x, mean, sigma):
        """caculate gauss density"""

        return 1.0 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    def generate_data(self, data_num):
        """generate multi-demensional gauss data"""

        sigma = 3
        data_one = np.random.multivariate_normal([10, 10], [[sigma, 0], [0, sigma]], data_num)
        data_two = np.random.multivariate_normal([20, 20], [[sigma, 0], [0, sigma]], data_num)
        data_three = np.random.multivariate_normal([30, 30], [[sigma, 0], [0, sigma]], data_num)
        gauss_data = data_one, data_two, data_three
        initial_data = []
        for data_set in gauss_data:
            data_set = [list(point) for point in data_set]
            for point in data_set:
                point += list(np.random.normal(0, 1, 5))
            initial_data += data_set
        return np.array(initial_data)

    def reduce_dimensionality(self, initial_data, target_dimension):
        """reduce dimensionaily"""

        dim = len(initial_data[0])
        mean = sum(initial_data) / len(initial_data)    # caculate mean
        Y = np.matrix(initial_data - mean)              # subtract mean
        cov_matrix = Y.T * Y / len(initial_data)        # covariance matrix
        feature_value, feature_vector = np.linalg.eig(cov_matrix)   # feature value and feacture vector
        value_vector = list(zip(feature_value, feature_vector))     # find D max feature value and vector
        max_value = heapq.nlargest(target_dimension, enumerate(value_vector), key=lambda x: x[1])
        choosed_vector = np.array([item[1][1] for item in max_value]).reshape(dim, target_dimension)
        if DEBUG: print('特征值\n', list(feature_value))
        return Y * choosed_vector  # transform dimensionality

    def cluster(self, converted_data):
        """cluster based on the converted data"""

        lab = GMM(epsilon=1e-5)
        initial_alpha = np.array([1 / 3, 1 / 3, 1 / 3])
        initial_means = np.array([-10.0, 0.0, 10.0])
        initial_sigma = np.array([1.0, 1.0, 1.0])
        parameters, iter_cnts = lab.EM(converted_data, initial_alpha, initial_means, initial_sigma, 100)
        if DEBUG: print(parameters)

        classified = []
        alpha, means, sigma = parameters
        for point in converted_data:
            probs = [alpha[k] * self.gauss_density(point, means[k], sigma[k]) for k in range(3)]
            classified.append(probs.index(max(probs)))
        return classified

    def display_cluster_result(self, initial_data, classified):
        """display result of cluster"""

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        xpts = [point[0] for point in initial_data]
        ypts = [point[1] for point in initial_data]
        plt.scatter(xpts, ypts, c='b')

        plt.subplot(1, 2, 2)
        xpts = [point[0] for point in initial_data]
        ypts = [point[1] for point in initial_data]
        plt.scatter(xpts, ypts, c=classified)
        plt.savefig('pca_for_cluster.jpg')
        if DEBUG: plt.show()

    def img2vector(self, filename):
        """convert image into vector"""

        img = cv2.imread(filename, 0)
        global img_shape
        img_shape = img.shape
        img_vector = np.reshape(img, (1, img_shape[0] * img_shape[1]))[0]
        return img_vector

    def load_dataset(self, folder):
        """load image and generate vectors"""

        files = sorted(os.listdir(folder), key=lambda x: int(x[:-4]))
        img_vectors = [self.img2vector(folder + '/' + filename) for filename in files]
        return [filename[:-4] for filename in files], img_vectors

    def pca_and_rebuild(self, initial_data, target_dimension):
        """reduce dimensionaily"""

        mean = sum(initial_data) / len(initial_data)
        Y = np.matrix(initial_data - mean)
        U, D, V = np.linalg.svd(Y)

        D = D ** 2  # compute feature value by singular value
        noise_signal_ratio = sum(D[:target_dimension]) / sum(D[target_dimension:])
        print('信噪比：', noise_signal_ratio)  # caculate noise-signal ratio

        feature_vector = V.T[:, :target_dimension]
        converted_data = Y * feature_vector
        rebuilt_data = converted_data * feature_vector.T + mean
        return feature_vector, converted_data, rebuilt_data

    def rebuild_img(self, rebuild_data, target_folder):
        """rebuild image and save into file"""

        for i in range(len(rebuild_data)):
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            target_path = target_folder + '/' + str(i + 1) + '.jpg'
            Image.fromarray(np.array(rebuild_data)[i].reshape(img_shape)).convert('RGB').save(target_path, 'JPEG')

    def display_rebuilt_result(self, src_folder, des_folder):
        """display the original imgs and rebuilt imgs"""

        src_files = sorted(os.listdir(src_folder), key=lambda x: int(x[:-4]))
        des_files = sorted(os.listdir(des_folder), key=lambda x: int(x[:-4]))
        src_imgs = [image.imread(src_folder + '/' + filename) for filename in src_files]
        des_imgs = [image.imread(des_folder + '/' + filename) for filename in des_files]

        plt.figure()
        for i in range(len(des_imgs) // 2):
            plt.subplot(4, 5, i + 1)
            plt.imshow(src_imgs[i])
            plt.axis('off')
            plt.subplot(4, 5, 10 + i + 1)
            plt.imshow(des_imgs[i])
            plt.axis('off')
        plt.savefig('pca_for_face.jpg')
        if DEBUG: plt.show()


if __name__ == '__main__':
    lab = PCA()

    # pca for multi-dimensional data cluster
    initial_data = lab.generate_data(50)
    converted_data = lab.reduce_dimensionality(initial_data, 1)
    converted_data = np.array([item[0, 0] for item in converted_data])
    # if DEBUG: print(converted_data)
    classified = lab.cluster(converted_data)
    lab.display_cluster_result(initial_data, classified)

    # pca for face dimensionalty reduction
    img_names, img_vectors = lab.load_dataset(folder='lfw')
    dataset = list(zip(img_names, img_vectors))
    feature_vector, converted_data, rebuilt_data = lab.pca_and_rebuild(img_vectors, 10)
    lab.rebuild_img(rebuilt_data, 'tmp')
    lab.display_rebuilt_result('lfw', 'tmp')
