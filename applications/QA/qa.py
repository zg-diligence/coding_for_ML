"""
  Copyright(c) 2018 Gang Zhang
  All rights reserved.
  Author: Gang Zhang
  Creation Date: 2018.4.13
  Last Modified: 2018.4.13

  Function:
        QA
"""

import os
import re
from collections import Counter
from string import punctuation as env_punc
from zhon.hanzi import punctuation as chs_punc
from pyltp import Segmentor, Postagger, NamedEntityRecognizer

DEBUG = True
puncs = env_punc + chs_punc
LTP_DATA_DIR = os.path.join(os.getcwd(), 'ltp_model')
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')


class QA(object):
    def __init__(self):
        pass

    def adjust_source_data(self, train_path, answer_path):
        """
        adjust source date, including remove unused words and do segmentation again
        :param train_path: path of train data
        :param answer_path: path of answer file
        :return: contents, questions, answers
        """

        segmentor = Segmentor()
        segmentor.load(cws_model_path)

        # read content
        train_text = open(train_path).read().strip()
        contents = re.split('<qid.+?\n', train_text)
        new_contents = []
        for content in contents:
            sentences = content.split('\n')
            new_sentences = []
            for sentence in sentences:
                words = sentence.split(' ')[2:-1]
                join_words = ''.join(words)
                words = list(segmentor.segment(join_words)) # do segmentation again
                words = [word for word in words if word not in puncs or word in ['：', '，']]
                if words:
                    new_sentences.append(words)
            new_contents.append(new_sentences)
        contents = new_contents

        # read questions
        train_text = re.sub('哪 国', '哪个 国家', train_text)
        questions = [line for line in train_text.split('\n') if '<qid' in line]
        for i in range(len(questions)):
            tmp = questions[i].split(' ')[2:-1]
            join_words = ''.join(tmp)
            tmp = segmentor.segment(join_words) # do segmentation again
            tmp = [word for word in tmp if word not in puncs or word in ['：', '，']]
            questions[i] = tmp
        segmentor.release()

        # read answers
        answers = open(answer_path).read().strip().split('\n')
        answers = [item.split( )[-1] for item in answers]

        with open('train.txt', 'w') as fw:
            fw.write('\n')
            for index, content in enumerate(contents):
                fw.write('text_' + str(index) + ':' + ' '.join(questions[index]) + '\n')
                for sentence in content:
                    fw.write(' '.join(sentence) + '\n')
                fw.write('\n')

        with open('questions.txt', 'w') as fw:
            for question in questions:
                fw.write(' '.join(question) + '\n')

        with open('answers.txt', 'w') as fw:
            fw.write('\n'.join(answers))

    def read_adjust_data(self, train_path, question_path, answer_path):
        """
        read adjust data, includign source text, questions, answers
        :param train_path: file path of train_data
        :param question_path: file path of questions
        :param answer_path: file path of answer
        :return: content, questions, answers
        """

        train_text = open(train_path).read().strip()
        contents = re.split('\ntext_.+?\n', train_text)
        contents = [[sent.split(' ') for sent in content.strip().split('\n')] for content in contents]
        contents[0] = contents[0][1:]

        questions = open(question_path).read().strip().split('\n')
        questions = [item.split(' ') for item in questions]

        answers = open('answers.txt').read().strip().split('\n')
        return contents, questions, answers

    def classify_questions(self, questions):
        """
        classify questions by 6W1H1O
        :param questions: all questions, already do segementation
        :return:
        """

        categories = ['WHAT', 'WHO', 'WHERE', 'WHEN', 'WHY', 'WHICH', 'HOW', 'OTHER']
        cate_ques = dict(zip(categories, [[] for _ in range(len(categories))]))
        question_cate = []
        for question in questions:
            tmp_ques = ''.join(question)
            if re.search('谁|什么人', tmp_ques):
                cate_ques['WHO'].append(question)
                question_cate.append('WHO')
            elif re.search('什么时间|什么时候|哪天', tmp_ques):
                cate_ques['WHEN'].append(question)
                question_cate.append('WHEN')
            elif re.search('哪.?个|哪座|哪件|哪种|哪位|哪边', tmp_ques):
                cate_ques['WHICH'].append(question)
                question_cate.append('WHICH')
            elif re.search('什么地方|哪里|哪儿|到哪|去哪|是哪|在哪|进哪|到了哪', tmp_ques):
                cate_ques['WHERE'].append(question)
                question_cate.append('WHERE')
            elif re.search('什么|干嘛', tmp_ques):
                cate_ques['WHAT'].append(question)
                question_cate.append('WHAT')
            elif re.search('怎么|怎样|如何|多少', tmp_ques):
                cate_ques['HOW'].append(question)
                question_cate.append('HOW')
            else:
                cate_ques['OTHER'].append(question)
                question_cate.append('OTHER')

        # if DEBUG:
        #     cate = categories[0]
        #     for ques in cate_ques[cate]:
        #         print(''.join(ques))
        #     print(len(cate_ques[cate]))

        return question_cate, cate_ques


if __name__ == '__main__':
    handler = QA()
    # handler.adjust_source_data('train.doc_query', 'reference.answer')
    contents, questions, answers = handler.read_adjust_data('train.txt', 'questions.txt', 'answers.txt')
    categories, cate_ques = handler.classify_questions(questions)

    answers = open('answers.txt').read().strip().split('\n')
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    postags = [postagger.postag([item, ])[0] for item in answers]
    # print(sorted(dict(Counter(postags)).items(), key=lambda item: item[1], reverse=True))
    tuples = list(zip(questions, categories, answers, postags))

    tmp_ques = []
    tmp_cate = []
    for item in tuples:
        if item[-1] == 'v':
            print(item)
            tmp_ques.append(''.join(item[0]))
            tmp_cate.append(item[1])
    # print(dict(Counter(tmp_cate)))

    for ques in tmp_ques:
        if not re.search('做什么|干什么|干嘛', ques):
            print(ques)

    # print(sum([1 for ques in tmp_ques if re.search('做什么|干什么', ques)]))
