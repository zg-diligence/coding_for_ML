"""
  Copyright(c) 2017 Gang Zhang
  All rights reserved.
  Author:Gang Zhang
  Date:2017.12.01
"""

import re
import math
import jieba
import codecs
from libsvm.svmutil import *
from collections import Counter
from itertools import chain, zip_longest

import os
DEBUG = False
DIR = os.getcwd()


class Preprocess(object):
    def __init__(self, stop_words, sample_num):
        self.stop_words = stop_words
        self.sample_num = sample_num

    def is_useful(self, word):
        """
        check whether if one word is a stop word
        """

        return word not in self.stop_words and not re.search('([A-Za-z]+|\d+\.*\d*)', word)

    def pos_one_text(self, file_path):
        """
        pos for one text and return the result as a list
        """

        with codecs.open(file_path, encoding='utf8') as fr:
            text = re.sub('(\s|\\u3000|\\x00|\\ue584)', '', fr.read())
            seg_list = list(jieba.cut(text, cut_all=False))
            return [item for item in seg_list if self.is_useful(item)]

    def pos_all_text(self, src_root_catelogue, des_root_catelogue, cnt_root_catelogue):
        """
        pos for all the texts
        count word frequency in one text
        write pos and cnt results into files
        """

        all_folders = os.listdir(src_root_catelogue)
        for folder in all_folders:
            if DEBUG: print(folder)
            src_folder = src_root_catelogue + '/' + folder
            des_folder = des_root_catelogue + '/' + folder
            cnt_folder = cnt_root_catelogue + '/' + folder
            if not os.path.exists(des_folder): os.mkdir(des_folder)
            if not os.path.exists(cnt_folder): os.mkdir(cnt_folder)
            files = sorted(os.listdir(src_folder), key=lambda item: int(item[:-4]))

            for i in range(len(files)):
                pos_result = self.pos_one_text(src_folder + '/' + files[i])
                cnt_words = [key + '\t' + str(value) for key, value in Counter(pos_result).items()]
                with codecs.open(des_folder + '/' + files[i], 'w', encoding='utf8') as fw:
                    fw.write('\n'.join(pos_result))
                with codecs.open(cnt_folder + '/' + files[i], 'w', encoding='utf8') as fw:
                    fw.write('\n'.join(cnt_words))

    def read_cnt_data(self, root_catelogue):
        """read words in each category, storaged in ndarray"""

        unique_category_words, all_category_words = [], []
        all_folders = os.listdir(root_catelogue)
        for foldername in all_folders:
            category_words = []
            cnt_folder = root_catelogue + '/' + foldername
            files = sorted(os.listdir(cnt_folder), key=lambda item: int(item[:-4]))[:self.sample_num]

            for filename in files:
                with codecs.open(cnt_folder + '/' + filename, encoding='utf8') as fr:
                    category_words.append([line.strip().split('\t')[0] for line in fr.readlines()])
            all_category_words.append(category_words)
            unique_category_words.append(set(list(chain(*category_words))))
        return unique_category_words, all_category_words

    def extract_feature_words(self, root_catelogue):
        """
        extract the first 1000 words in every category as feature words and write them into files
        """

        unique_category_words, all_category_words = self.read_cnt_data(root_catelogue)
        cnt_category_words = [dict(zip_longest(words, [], fillvalue=0)) for words in unique_category_words]
        category_num = len(all_category_words)
        category_file_num = len(all_category_words[0])
        total_file_num = category_num * category_file_num

        for i in range(category_num):
            for file_words in all_category_words[i]:
                for word in file_words:
                        cnt_category_words[i][word] += 1

        all_feature_words = []
        for i in range(category_num):
            if DEBUG: print('category_%d' % (i + 1))
            chi_values = []
            for word in unique_category_words[i]:
                a = cnt_category_words[i][word]
                b = sum([cnt_category_words[j].get(word, 0) for j in range(category_num)])
                c = category_file_num - a
                d = category_file_num * (category_num - 1) - b
                chi_values.append((a*d - b*c)**2 * total_file_num / ((a+b) * (a+c) * (b+d) * (c+d)))

            sort_key = lambda item: (item[1], item[0])  # sorted by primary key and secondary key
            word_chi_values = sorted(list(zip(unique_category_words[i], chi_values)), key=sort_key, reverse=True)
            category_feature_words = [word for word, _ in word_chi_values[:1000]]
            all_feature_words.extend(category_feature_words)
            with codecs.open('data_tmp/category_feature_' + str(i+1) + '.txt', 'w', encoding='utf8') as fw:
                fw.write('\n'.join(category_feature_words))

        all_feature_words = sorted(set(all_feature_words))
        words_idf_values = self.caculate_IDF(total_file_num, all_category_words, all_feature_words)
        with codecs.open('feature_words.txt', 'w', encoding='utf8') as fw:
            extract_features = sorted(words_idf_values, key=lambda item: item[1], reverse=True)
            extract_features = ['{:<10d}{:10}\t{:<10f}'.format(index + 1, word, idf) for index, (word, idf) in enumerate(extract_features)]
            fw.write('\n'.join(extract_features))

    def caculate_IDF(self, total_file_num, all_category_words, feature_words):
        """caculate words idf value"""

        word_text_cnts = {}
        for category_words in all_category_words:
            for file_words in category_words:
                for word in file_words:
                    if not word_text_cnts.get(word):
                        word_text_cnts[word] = 1
                    else:
                        word_text_cnts[word] += 1
        idf_vales = [math.log(total_file_num / (word_text_cnts[word] + 1)) for word in feature_words]
        return list(zip(feature_words, idf_vales))

    def generate_text_vector(self, root_catelogue):
        """
        convert all the texts into vectors and write them into file
        """

        with codecs.open('feature_words.txt', encoding='utf8') as fr:
            feature_words = [re.split('\s+', line.strip()) for line in fr.readlines()]
        feature_words = dict([(word[1], (word[0], word[2])) for word in feature_words])

        all_folders = os.listdir(root_catelogue)
        category_feature_words = []
        for i in range(1, len(all_folders)+1):
            with codecs.open('data_tmp/category_feature_' + str(i) + '.txt', encoding='utf8') as fr:
                category_feature_words.append([line.strip() for line in fr.readlines()])

        train_vectors, test_vectors = [], []
        for i in range(len(all_folders)):
            if DEBUG: print('category_%d' % (i + 1))
            category_vectors = []
            cnt_folder = root_catelogue + '/' + all_folders[i]
            files = sorted(os.listdir(cnt_folder), key=lambda item: int(item[:-4]))[:self.sample_num]
            for filename in files:
                with codecs.open(cnt_folder + '/' + filename, encoding='utf8') as fr:
                    vector = []
                    for word, num in [line.strip().split('\t') for line in fr.readlines()]:
                        if word in category_feature_words[i]:
                            TF_IDF = float(num) * float(feature_words[word][1])
                            vector.append((int(feature_words[word][0]), TF_IDF))
                    if len(vector) < 5: continue
                    vector = sorted(vector, key=lambda item: item[0])
                    vector = [str(item[0]) + ':' + str(item[1]) for item in vector]
                    category_vectors.append(str(i+1)+' ' + ' '.join(vector))
            train_vectors += category_vectors[:len(category_vectors) * 9 //10]
            test_vectors += category_vectors[ len(category_vectors) * 9 //10:]

        with codecs.open('train_scale', 'w', encoding='utf8') as fw:
                fw.write('\n'.join(train_vectors))
        with codecs.open('test_scale', 'w', encoding='utf8') as fw:
                fw.write('\n'.join(test_vectors))


class SVM(object):
    def __init__(self):
        pass

    def train(self, data_path, model_path, param_str=''):
        """train model and save it into file"""

        if not param_str:
            param_str = '-c 200 -g 9.0e-6 -q'

        y, x = svm_read_problem(data_path)
        svm_prob = svm_problem(y, x)
        svm_param = svm_parameter(param_str)
        model = svm_train(svm_prob, svm_param)
        svm_save_model(model_path, model)

    def predict(self, data_path, modal_path):
        """predict label for unlabel data"""

        y, x = svm_read_problem(data_path)
        svm_model = svm_load_model(modal_path)
        p_label, p_acc, p_val = svm_predict(y, x, svm_model)
        return p_label, p_acc, p_val


def run_preprocess(sample_num = 100):
    """
    run all needed preprocess
    """

    with codecs.open('stop_words.txt', encoding='utf8') as fr:
        stop_words = list(fr.read().strip().split('\r\n'))

    src_root_catelogue = DIR + '/data_svm'  # original data
    des_root_catelogue = DIR + '/data_pos'  # pos data -- part of speech
    cnt_root_catelogue = DIR + '/data_cnt'  # cnt data -- word frequency
    tmp_root_catelogue = DIR + '/data_tmp'  # tmp data -- feature words of each category

    if not os.path.exists(src_root_catelogue):
        os.mkdir(src_root_catelogue)
    if not os.path.exists(des_root_catelogue):
        os.mkdir(des_root_catelogue)
    if not os.path.exists(cnt_root_catelogue):
        os.mkdir(cnt_root_catelogue)
    if not os.path.exists(tmp_root_catelogue):
        os.mkdir(tmp_root_catelogue)

    lab = Preprocess(stop_words, sample_num)
    lab.pos_all_text(src_root_catelogue, des_root_catelogue, cnt_root_catelogue)
    lab.extract_feature_words(cnt_root_catelogue)
    lab.generate_text_vector(cnt_root_catelogue)


if __name__ == '__main__':
    run_preprocess(1000)

    svm_lab = SVM()
    param_str = '-c 200 -g 9.0e-6 -q'
    svm_lab.train('train_scale', 'svm.model', param_str)
    svm_lab.predict('test_scale', 'svm.model')
