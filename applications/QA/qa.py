"""
  Copyright(c) 2018 Gang Zhang
  All rights reserved.
  Author: Gang Zhang
  Creation Date: 2018.4.13
  Last Modified: 2018.4.14

  Function:
        QA
"""

import os
import re
from collections import Counter
from string import punctuation as env_punc
from zhon.hanzi import punctuation as chs_punc
from pyltp import Segmentor, Postagger, Parser

DEBUG = True
puncs = env_punc + chs_punc
LTP_DATA_DIR = os.path.join(os.getcwd(), 'ltp_model')
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')


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
                words = list(segmentor.segment(join_words))  # do segmentation again
                words = [word for word in words if word not in puncs or word in ['：', '，']]
                if words:
                    new_sentences.append(words)
            new_contents.append(new_sentences)
        new_contents[-1] = new_contents[-1][:-1]
        contents = new_contents

        # read questions
        train_text = re.sub('哪 国', '哪个 国家', train_text)
        questions = [line for line in train_text.split('\n') if '<qid' in line]
        for i in range(len(questions)):
            tmp = questions[i].split(' ')[2:-1]
            join_words = ''.join(tmp)
            tmp = segmentor.segment(join_words)  # do segmentation again
            tmp = [word for word in tmp if word not in puncs or word in ['：', '，']]
            questions[i] = tmp

        # read answers
        answers = open(answer_path).read().strip().split('\n')
        answers = [item.split()[-1] for item in answers]

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

        answers = open(answer_path).read().strip().split('\n')
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

        if DEBUG:
            cate = categories[1]
            # for ques in cate_ques[cate]:
            #     print(''.join(ques))
            # print(len(cate_ques[cate]))

        return question_cate, cate_ques

    def LCS(self, A, B):
        """
        to get the least common sequence
        :param A: list A
        :param B: list B
        :return: LCS
        """

        lengths = [[0 for _ in range(len(B) + 1)] for _ in range(len(A) + 1)]

        # row 0 and column 0 are initialized to 0 already
        for i, x in enumerate(A):
            for j, y in enumerate(B):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        x, y = len(A), len(B)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                y -= 1
            else:
                assert A[x - 1] == B[y - 1]
                result.append(A[x - 1])
                x -= 1
                y -= 1
        return result[::-1]

    def get_keywords(self, words):
        """
        get keywords of word list for a sentence
        :param words: word list for a sentence
        :return: keywords, set
        """

        postags = postagger.postag(words)
        arcs = parser.parse(words, postags)

        tmp = []
        for index, arc in enumerate(arcs):
            if arc.relation in ['VOB', 'SBV']:
                tmp.append(words[index])
                tmp.append(words[arc.head - 1])
            elif arc.relation == 'HED':
                tmp.append(words[index])

        for index, tag in enumerate(postags):
            if tag == 'v':
                tmp.append(words[index])

        return set(tmp) - {'什么', '谁', '哪儿', '是'}

    def get_sent_by_keywords(self, ques, text):
        """
        get the most similar sentence by extracting keywords
        :param ques: questions
        :param text: text/story
        :return: rate, sentence
        """

        keywords = sorted(self.get_keywords(ques))
        postags = postagger.postag(list(keywords))
        if not keywords: return 0, None

        max_cnt = 0
        max_sent = None
        max_is_verbs = False
        for sent in text:
            cnt = 0
            is_verbs = False
            for index, word in enumerate(keywords):
                if word in sent:
                    cnt += 1
                    if postags[index] == 'v':
                        is_verbs = True
            if cnt > max_cnt:
                max_cnt = cnt
                max_sent = sent
                max_is_verbs = is_verbs

        rate = max_cnt / len(keywords)
        rate = 0 if not max_is_verbs and rate < 0.7 else rate
        return rate, max_sent,

    def get_sent_by_lcs(self, ques, text):
        """
        get the most similar sentence by extracting LCS
        :param ques: question
        :param text: text/story
        :return: rate, sentence
        """

        max_length, max_sent = 0, None
        for sent in text:
            lcs = self.LCS(ques, sent)
            if len(lcs) > max_length:
                max_length = len(lcs)
                max_sent = sent
        rate = max_length / len(ques)
        return rate, max_sent

    def get_ans_sents(self, contents, questions):
        """
        get the most similar sentences all questions
        :param contents: all texts/stories
        :param questions: all questions
        :return: all extracted sentences
        """

        max_sents = []
        cnt = [0 for _ in range(7)]
        fw = open('failed_matched_sents.txt', 'w')
        for ques, text in zip(questions, contents):
            rate_1, max_sent_1 = self.get_sent_by_lcs(ques, text)             # LCS based on phrase
            if rate_1 >= 0.6:
                cnt[0] += 1
                max_sents.append([max_sent_1,])
            else:
                tmp_ques = ''.join(ques)
                tmp_text = [''.join(sent) for sent in text]
                rate_2, max_sent_2 = self.get_sent_by_lcs(tmp_ques, tmp_text) # LCS based on word
                max_sent_2 = text[tmp_text.index(max_sent_2)]
                if  rate_2 >= 0.6:
                    cnt[1] += 1
                    max_sents.append([max_sent_2,])
                else:
                    tmp_rate_3, max_sent_3 = self.get_sent_by_keywords(ques, text)  # based on keywords
                    if tmp_rate_3 >= 0.6:
                        cnt[2] += 1
                        max_sents.append([max_sent_3, max_sent_2]) # mainly max_sent_3
                    else:
                        cnt[3] += 1
                        if not max_sent_1:
                            max_sent_1 = ['None']
                        if not max_sent_2:
                            max_sent_2 = ['None']
                        if not max_sent_3:
                            max_sent_3 = ['None']
                        max_sents.append([max_sent_2, max_sent_1, max_sent_3]) # mainly max_sent_2

                        index = questions.index(ques)
                        fw.write('text_' + str(index) + ':' + ' '.join(ques) + '\n')
                        fw.write(''.join(max_sent_1) + '\n' + ''.join(max_sent_2) + '\n' + ''.join(max_sent_3) + '\n\n')
                        for sentence in text:
                            fw.write(' '.join(sentence) + '\n')
                        fw.write('\n')

        if DEBUG:print(sum(cnt[:3]), cnt)
        with open('ques_matched_sents.txt', 'w') as fw:
            for i in range(len(questions)):
                ques = questions[i]
                sents = max_sents[i]
                fw.write('text_' + str(i) + ':' + ' '.join(ques) + '\n')
                for sent in sents:
                    fw.write(' '.join(sent) + '\n')
                fw.write('\n')
        return max_sents

    def count_ans_pos(self, ans):
        """
        count pos of answers
        :param ans: list of answers
        :return: pos of all answers
        """

        postags = [postagger.postag([item, ])[0] for item in ans]
        cnt_postags = sorted(dict(Counter(postags)).items(), key=lambda item: item[1], reverse=True)
        if DEBUG:
            for item in cnt_postags:
                print(item[0], '\t=>\t', item[1])
        return postags

    def read_extract_sents(self, file_path):
        """
        read extracted sentence for each questions
        :param file_path:
        :return: questions, exracted_sents
        """

        with open(file_path) as fr:
            text = [story.split('\n') for story in fr.read().strip().split('\n\n')]
            questions = [story[0].split(':')[1].split(' ') for story in text]
            extract_sents = [[line.split(' ') for line in story[1:]] for story in text]
            return questions, extract_sents

    def imp_LCS(self, A, B):
        """
        to get the least common sequence
        :param A: list A
        :param B: list B
        :return: LCS
        """

        lengths = [[0 for _ in range(len(B) + 1)] for _ in range(len(A) + 1)]

        # row 0 and column 0 are initialized to 0 already
        for i, x in enumerate(A):
            for j, y in enumerate(B):
                if x == y:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

        # read the substring out from the matrix
        result = []
        x, y = len(A), len(B)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                y -= 1
            else:
                assert A[x - 1] == B[y - 1]
                result.append((A[x - 1], y-1))
                x -= 1
                y -= 1
        return result[::-1]

    def extract_ans_for_WHO(self, questions, categories, extracted_sents, pred_answers):
        cnt = 0
        sbv_cnt = 0
        vob_cnt = 0
        for i in range(len(questions)):
            if categories[i] != 'WHO':
                continue

            ques = [word for word in questions[i] if word not in stop_words]
            sent = [word for word in extracted_sents[i][0] if word not in stop_words]

            if '谁' not in ques:
                continue

            pos = ques.index('谁')
            ques[pos] = '小明'
            ques_tag = list(postagger.postag(ques))
            ques_rel = list(parser.parse(ques, ques_tag))
            sent_tag = list(postagger.postag(sent))
            sent_rel = list(parser.parse(sent, sent_tag))
            # nouns = [ques[i] for i in range(len(ques)) if 'n' in ques_tag[i] and ques[i] != '小明']

            lcs = self.imp_LCS(ques, sent)
            if not lcs: continue

            if pos == 0:
                index = lcs[0][1]
            elif pos == len(ques)-1:
                index = lcs[-1][1]
            else:
                lindex = 0
                rindex = 0
                for item in lcs:
                    if ques.index(item[0]) < pos:
                        lindex = item[1]
                    elif ques.index(item[0]) > pos:
                        rindex = item[1]
                        break
                print(answers[i])
                print(lcs)
                print(ques)
                print(sent)
                print(sent[lindex+1:rindex])

            # index = ques.index('小明')
            # if ques_rel[index].relation == 'SBV':
            #     sbv_verb = ques[ques_rel[index].head-1]
            #     for word, index in lcs:
            #         flag = False
            #         if word == sbv_verb:
            #             for j in range(index-1, -1, -1):
            #                 if 'n' in sent_tag[j] and sent_rel[j].relation == 'SBV' and sent[j] not in nouns:
            #                     sbv_cnt += 1
            #                     if answers[i] !=  sent[j]:
            #                         print(ques)
            #                         print(sent)
            #                         print(lcs)
            #                         print(sbv_verb, answers[i], sent[j])
            #                         print()
            #                     else:
            #                         cnt += 1
            #                     flag = True
            #                     break
            #         if flag: break
            # elif ques_rel[index].relation == 'VOB':
            #     vob_cnt += 1

        print(sbv_cnt, vob_cnt, cnt)

    # re.search('干什么|做什么|玩什么|干嘛|学什么|学会什么|什么比赛|比赛\w*什么|受到什么', join_ques):
    # not re.search('想做什么|叫做什么|称做什么|干什么的', join_ques):


if __name__ == '__main__':
    parser = Parser()
    segmentor = Segmentor()
    postagger = Postagger()
    parser.load(par_model_path)
    segmentor.load_with_lexicon(cws_model_path, 'outer_dict.txt')
    postagger.load(pos_model_path)

    with open('stop_words.txt') as fr:
        stop_words = fr.read().strip().split('\n')

    handler = QA()
    handler.adjust_source_data('train.doc_query', 'reference.answer')
    contents, questions, answers = handler.read_adjust_data('train.txt', 'questions.txt', 'answers.txt')
    extracted_sents = handler.get_ans_sents(contents, questions)
    # postags = handler.count_ans_pos(answers)

    for question in questions:
        print(question)

    # pred_answers = ['' for _ in range(len(questions))]
    # questions, extracted_sents = handler.read_extract_sents('ques_matched_sents.txt')
    # categories, cate_ques = handler.classify_questions(questions)
    # handler.extract_ans_for_WHO(questions, categories, extracted_sents, pred_answers)

    parser.release()
    segmentor.release()
    postagger.release()
