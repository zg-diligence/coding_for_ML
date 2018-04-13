"""
  Copyright(c) 2017 Gang Zhang
  All rights reserved.
  Author:Gang Zhang
  Date:2017.12.22
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DEBUG = True
SUBMISSION = True
pca_components = 15
plt.rcParams['font.sans-serif']=['SimHei']
col_names = ['wdith', 'height', 'width/height', 'width*height', 'ishorizontal']
contour_names = ['area', 'perimeter', 'centroidx', 'centroidy', 'converxity', 'len_hull', 'sum_hull', 'len_approx']

class PreProcess(object):
    def __init__(self):
        pass

    def resize_img(self, src_folder, des_folder):
        """
        resize picture in the given folder
        :param src_folder: source folder path
        :param des_folder: destination folder path
        :return:
        """

        if not os.path.exists(des_folder):
            os.mkdir(des_folder)

        files = sorted(os.listdir(src_folder), key=lambda item: int(item[:-4]))
        for filename in files:
            img = Image.open(src_folder+'/'+filename)
            img = img.resize((50, 50), Image.ANTIALIAS)
            img.save(des_folder + '/' + filename, quality=100)

    def extract_basic_features(self, folder):
        """
        extract width, height, ratio, square, ishor
        :param folder: source folder path
        :return:
        """

        features = []
        files = sorted(os.listdir(folder), key=lambda item: int(item[:-4]))
        for filename in files:
            img = Image.open(folder + '/' + filename)
            width, height = img.size
            ratio = width / height
            square = width * height
            ishor = int(width > height)
            features.append((width, height, square, ratio, ishor))
        return features

    def extract_pca_features(self, folder, n_components):
        """
        extract PCA feature
        :param folder: source folder path
        :param n_components: number of dimension to keep
        :return:
        """

        pca = PCA(n_components=n_components, svd_solver='full')
        files = sorted(os.listdir(folder), key=lambda x: int(x[:-4]))
        img_vectors = []
        for filename in files:
            img = Image.open(folder + '/' + filename)
            img_shape = img.size
            img_vector = np.reshape(img, (1, img_shape[0] * img_shape[1]))[0]
            img_vectors.append(img_vector)
        return pca.fit_transform(img_vectors)

    def extract_contour_features(self, folder):
        """
        extract contour features
        :param folder: source folder path
        :return:
        """

        files = sorted(os.listdir(folder), key=lambda x: int(x[:-4]))
        features = []
        for filename in files:
            img = cv2.imread(folder+'/'+filename, 0)
            _, thresh = cv2.threshold(img, 127, 255, 0)
            _, contours, _ = cv2.findContours(thresh, 1, 2)

            cnt = contours[0]
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            convexity = int(cv2.isContourConvex(cnt))

            hull = cv2.convexHull(cnt, returnPoints=False)
            sum_hull = sum([item[0] for item in hull])
            approx = cv2.approxPolyDP(cnt, perimeter / 100, True)
            moments = cv2.moments(cnt)
            moments_values = list(moments.values())

            try:
                centroidx = moments['m10'] / moments['m00']
                centroidy = moments['m01'] / moments['m00']
            except:
                centroidx = 0
                centroidy = 0

            item_features = [area, perimeter, centroidx, centroidy, convexity,
                             len(hull), sum_hull, len(approx)] + moments_values
            features.append(item_features)
        return features

def run_preprocess(training_path, testing_path, src_images, des_images):
    """
    feature engineering, extracting 52 extra features
    :param training_path: path of training-set
    :param testing_path: path of testing-set
    :param src_images: source folder path of images
    :param des_images: destionation folder path of images
    :return:
    """

    lab = PreProcess()

    print('compressing_picture...')
    lab.resize_img(src_images, des_images)

    train_data = pd.read_csv(training_path)
    test_data = pd.read_csv(testing_path)

    # add basic features
    print('extracting basic features...')
    start_index = train_data.shape[1]
    basic_features = lab.extract_basic_features(src_images)
    for i in range(len(col_names)):
        col_data = [basic_features[index - 1][i] for index in train_data.id]
        train_data.insert(start_index + i, col_names[i], col_data)
        col_data = [basic_features[index - 1][i] for index in test_data.id]
        test_data.insert(start_index + i - 1, col_names[i], col_data)

    # add contour features
    print('extracting contour features...')
    start_index += len(col_names)
    contour_features = lab.extract_contour_features(src_images)
    for i in range(len(contour_names)):
        col_data = [contour_features[index - 1][i] for index in train_data.id]
        train_data.insert(start_index + i, contour_names[i], col_data)
        col_data = [contour_features[index - 1][i] for index in test_data.id]
        test_data.insert(start_index + i - 1, contour_names[i], col_data)

    for i in range(len(contour_names), len(contour_features[0])):
        col_data = [contour_features[index - 1][i] for index in train_data.id]
        train_data.insert(start_index + i, 'moments' + str(i + 1 - len(contour_names)), col_data)
        col_data = [contour_features[index - 1][i] for index in test_data.id]
        test_data.insert(start_index + i - 1, 'moments' + str(i + 1 - len(contour_names)), col_data)

    # add pca features
    print('extracting pca features...')
    start_index += len(contour_features[0])
    pca_features = lab.extract_pca_features(des_images, n_components=pca_components)

    transform_data = [pca_features[index - 1] for index in train_data.id]
    transform_data = np.matrix(transform_data).T
    for i in range(pca_components):
        lis = transform_data[i].tolist()[0]
        train_data.insert(start_index + i, 'pca' + str(i + 1), lis)

    transform_data = [pca_features[index - 1] for index in test_data.id]
    transform_data = np.matrix(transform_data).T
    for i in range(pca_components):
        lis = transform_data[i].tolist()[0]
        test_data.insert(start_index + i - 1, 'pca' + str(i + 1), lis)

    # write into csv files
    with open('new_'+training_path, 'w') as fw:
        fw.write(train_data.to_csv(index=None))
    with open('new_'+testing_path, 'w') as fw:
        fw.write(test_data.to_csv(index=None))

def compare_methods(train_path, train_num=600):
    """
    compare accuracy and logloss among methods
    :param train_path: path of the training-set
    :param train_num: number of samples for training
    :return:
    """

    methods = ['LR', 'LDA', 'SVM', 'Bayes', 'KNN', 'RF', 'DT', 'GB']

    classifiers = [
        LogisticRegression(solver='lbfgs', multi_class="multinomial", C=20000, tol=1e-6),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.8),
        SVC(kernel="linear", C=1.0, probability=True),
        BernoulliNB(alpha=0.5),
        KNeighborsClassifier(2, algorithm='kd_tree'),
        RandomForestClassifier(n_estimators=22, criterion='entropy', max_depth=10, random_state=0),
        DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0),
        GradientBoostingClassifier(n_estimators=80),
    ]

    data = pd.read_csv(train_path)
    data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data)

    scores, losses = [], []
    for model in classifiers:
        model.fit(X[:train_num, ], y[:train_num])
        score = model.score(X[train_num:], y[train_num:])
        probs = model.predict_proba(X[train_num:])
        losses.append(log_loss(y[train_num:],probs, labels=list(range(99))))
        scores.append(score)

    plt.figure()
    plt.title('不同方法准确率对比')
    plt.xlabel('methods')
    plt.ylabel('accuracy')
    plt.xticks(list(range(0, 8, 1)), methods)
    plt.bar(range(len(scores)), scores)
    for a, b in zip(list(range(len(scores))), scores):
        plt.text(a, b, '%.3f' %  b, ha='center', va='bottom', fontsize=14)
    plt.savefig('./tmp_result/compare_score.jpg')
    plt.close()

    plt.figure()
    plt.title('不同方法对数损失对比')
    plt.xlabel('methods')
    plt.ylabel('loss')
    plt.xticks(list(range(0, 8, 1)), methods)
    plt.bar(range(len(losses)), losses)
    for a, b in zip(list(range(len(losses))), losses):
        plt.text(a, b, '%.3f' %  b, ha='center', va='bottom', fontsize=14)
    plt.savefig('./tmp_result/compare_loss.jpg')
    plt.close()

def display_confused_images(confused_ids, confused_class):
    """
    display confused_species
    :param confused_ids: id of image belongs to the confused classes
    :param confused_class: all the confused classes
    :return:
    """

    imgs = [cv2.imread('./images/' + str(item_id) + '.jpg') for item_id in confused_ids]

    plt.figure(figsize=(16, 12))
    for i in range(len(imgs)):
        plt.subplot(len(imgs) // 10, 10, i+1)
        plt.imshow(imgs[i])
        if not i % 10:
            plt.title(confused_class[i//10], loc='left')
        plt.axis('off')
    plt.savefig('./tmp_result/confused_species.jpg')

def display_confused_probs(probs, name):
    """
    display predicted probablity for confused objects
    :param probs: predicted probs of confused object
    :param name: id of the confused image
    :return:
    """

    plt.figure()
    plt.xlabel('类别')
    plt.ylabel('概率')
    plt.title('id = ' + str(name))
    plt.bar(range(len(probs)), probs)
    plt.savefig('./tmp_result/confused_' + str(name) + '.jpg')

def run_for_confused(train_data, confused_classes, confused_items, param_C=1e10, param_tol=1e-10):
    """
    training with objects belongs to the confused classes and predict again
    :param train_data: training set, same as original traing set
    :param confused_classes: index of confused class from confused_items
    :param confused_items: the objects from testing set that cannot be predicted accuratly
    :param param_C: param C of LR
    :param param_tol: param tol og LR
    :return: index of predicted class
    """

    # find confused classes
    confused_class = []
    species = sorted(train_data.species.unique())
    for i in range(len(confused_classes)):
        confused_class += [species[j] for j in confused_classes[i]]
    confused_class = sorted(set(confused_class))

    # find training data belongs to the confused classes
    confused_data = []
    for specie in confused_class:
        confused_data += list(train_data.loc[train_data['species'] == specie].values)
    confused_data = pd.DataFrame(confused_data, columns=train_data.columns)

    # output information about confused data
    if DEBUG:
        print('confused_class =', len(confused_class),
              ', confused_objects =', len(confused_data),
              ', confused_items =', len(confused_items), '\n')

    # train and predict again
    confused_ids = confused_data.pop('id')
    if SUBMISSION: display_confused_images(confused_ids, confused_class)

    y = confused_data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(confused_data).transform(confused_data)
    model = LogisticRegression(solver='lbfgs', multi_class="multinomial", C=param_C, tol=param_tol)
    model.fit(X, y)
    ypred = model.predict(confused_items)
    ypred = [species.index(confused_class[index]) for index in ypred]
    probs = model.predict_proba(confused_items)

    return ypred, [max(prob) for prob in probs]

def run_training_set(train_path, train_num=600, param_C=20000, param_tol=1e-6):
    """
    :param train_path: path of training-set
    :param train_num: number of objects used for training
    :param param_C: param C of LR
    :param param_tol: param tol og LR
    :return:
    """

    # train and predict
    data = pd.read_csv(train_path)
    copy_data = data.copy()
    species = sorted(data.species.unique())

    data_id = data.pop('id')
    y = data.pop('species')
    y = LabelEncoder().fit(y).transform(y)
    X = StandardScaler().fit(data).transform(data)

    LR = LogisticRegression(solver='lbfgs', multi_class="multinomial", C=param_C, tol=param_tol)
    LR.fit(X[:train_num, ], y[:train_num])
    ypred = LR.predict_proba(X[train_num:])
    score = LR.score(X[train_num:], y[train_num:])

    # filter the confused objects
    threahold = 0.125
    correct_label = []
    confused_index = []
    confused_items = []
    confused_classes = []
    for i in range(len(ypred)):
        confused_class = [j for j in range(len(ypred[i])) if ypred[i][j] > threahold]
        if max(ypred[i]) < 1 - 2*threahold:
            confused_index.append(i)
            confused_classes.append(confused_class)
            confused_items.append(X[train_num + i])
            correct_label.append(y[train_num + i])

    # output infomation of confused objects
    if DEBUG:
        for i in range(len(ypred)):
            if max(ypred[i]) < 1 - 2 * threahold:
                classes = [j for j in range(len(ypred[i])) if ypred[i][j] > threahold]
                confused = [species[j] for j in range(len(ypred[i])) if ypred[i][j] > threahold]
                probs = [ypred[i][j] for j in range(len(ypred[i])) if ypred[i][j] > threahold]
                print('index =', i + 1, 'id =', data_id[train_num+i], '\t', classes, probs, confused)
        print()

    # train and predict for the confused objects
    ypred2, probs = run_for_confused(copy_data[:train_num], confused_classes, confused_items, 1e10, 1e-10)

    # output result of re-train for the confused objects
    if DEBUG:
        result = list(zip(correct_label, ypred2, probs))
        for line in result: print(line)
        print()

    # adjust the prediction probs, vote to set 0 or 1
    adjust_pred = []
    for i in range(len(ypred)):
        pred = ypred[i]
        res = [0 for _ in range(len(pred))]
        if i not in confused_index:
            res[list(pred).index(max(pred))] = 1
        else:
            pred_index = ypred2.pop(0)
            if probs.pop(0) > max(pred):
                res[pred_index] = 1
            else:
                res[list(pred).index(max(pred))] = 1
        adjust_pred.append(np.array(res))
    ypred = np.array(adjust_pred)

    loss = log_loss(y[train_num:], ypred, labels=list(range(99)))
    print('log_loss =', loss, ', accuracy =', score); print()

def run_submission(train_path, test_path, param_C=20000, param_tol=1e-6):
    """
    train with all the training-set and predict the testing-set
    :param train_path: path of training-set
    :param test_path: path of testing-set
    :param param_C: param C of LR
    :param param_tol: param tol og LR
    :return:
    """

    # train and predict
    train_data = pd.read_csv(train_path)
    copy_data = train_data.copy()
    species = sorted(train_data.species.unique())

    train_data.pop('id')
    train_y = train_data.pop('species')
    train_y = LabelEncoder().fit(train_y).transform(train_y)
    train_X = StandardScaler().fit(train_data).transform(train_data)

    test_data = pd.read_csv(test_path)
    test_id = test_data.pop('id')
    test_X = StandardScaler().fit(test_data).transform(test_data)

    LR = LogisticRegression(solver='lbfgs', multi_class="multinomial", C=param_C, tol=param_tol)
    LR.fit(train_X, train_y)
    ypred = LR.predict_proba(test_X)

    # filter the confused objects
    threahold = 0.05
    confused_index = []
    confused_items = []
    confused_classes = []
    for i in range(len(ypred)):
        confused_class = [j for j in range(len(ypred[i])) if ypred[i][j] > threahold]
        if max(ypred[i]) < 1 - 2 * threahold:
            confused_index.append(i)
            confused_classes.append(confused_class)
            confused_items.append(test_X[i])

    # output infomation of confused objects
    if DEBUG:
        for i in range(len(ypred)):
            if max(ypred[i]) < 1-2*threahold:
                classes = [j for j in range(len(ypred[i])) if ypred[i][j] > threahold]
                confused = [species[j] for j in range(len(ypred[i])) if ypred[i][j] > threahold]
                probs = [ypred[i][j] for j in range(len(ypred[i])) if ypred[i][j] > threahold]
                display_confused_probs(probs=ypred[i], name=test_id[i])
                print('index =', i+1, 'id =', test_id[i], '\t', classes, probs, confused)
        print()

    # train and predict for the confused objects
    ypred2, probs = run_for_confused(copy_data, confused_classes, confused_items, 1e10, 1e-12)
    if DEBUG: print(list(zip(ypred2, probs)))

    # adjust the prediction probs, vote to set 0 or 1
    adjust_pred = []
    for i in range(len(ypred)):
        pred = ypred[i]
        res = [0 for _ in range(len(pred))]
        if i not in confused_index:
            res[list(pred).index(max(pred))] = 1
        else:
            pred_index = ypred2.pop(0)
            if probs.pop(0) > max(pred):
                res[pred_index] = 1
            else:
                res[list(pred).index(max(pred))] = 1
        adjust_pred.append(np.array(res))
    ypred = np.array(adjust_pred)
    ypred = pd.DataFrame(ypred, index=test_id, columns=species)

    with open('submission.csv', 'w') as fw:
        fw.write(ypred.to_csv())

if __name__ == '__main__':
    train_path = 'new_train.csv'
    test_path = 'new_test.csv'
    param_C, param_tol = 20000, 1e-6

    if not os.path.exists('./tmp_result'):
        os.mkdir('./tmp_result')

    print('running preprocess...')
    run_preprocess('train.csv', 'test.csv', './images', './new_images')

    print('comapring methods...')
    # compare_methods('train.csv', 600)

    print('training and predicting the data-set...\n')
    if not  SUBMISSION:
        run_training_set(train_path, train_num=600, param_C=param_C, param_tol=param_tol)
    else:
        run_submission(train_path, test_path, param_C=param_C, param_tol=param_tol)
