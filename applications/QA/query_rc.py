"""
  Copyright(c) 2018 Gang Zhang
  All rights reserved.
  Author: Gang Zhang
  Creation Date: 2018.4.13
  Last Modified: 2018.4.21

  Function:
        User-query for reading comprehension
"""

import os
import re
from collections import Counter
from string import punctuation as env_punc
from zhon.hanzi import punctuation as chs_punc
from pyltp import Segmentor, Postagger, Parser

DEBUG = False
puncs = env_punc + chs_punc
TMP_RESULT = os.path.join(os.getcwd(), 'tmp_result')
LTP_DATA_DIR = os.path.join(os.getcwd(), 'ltp_model')
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')

des = ['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws']
rags = ['SBV', 'VOB', 'POB', 'IOB', 'FOB', 'DBL', 'ATT', 'ADV', 'CMP', 'COO', 'LAD', 'RAD', 'IS', 'WP']

class RC(object):
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

        with open(TMP_RESULT + '/train.txt', 'w') as fw:
            fw.write('\n')
            for index, content in enumerate(contents):
                fw.write('text_' + str(index) + ':' + ' '.join(questions[index]) + '\n')
                for sentence in content:
                    fw.write(' '.join(sentence) + '\n')
                fw.write('\n')

        with open(TMP_RESULT + '/train_ques.txt', 'w') as fw:
            for question in questions:
                fw.write(' '.join(question) + '\n')

        with open(TMP_RESULT + '/answers.txt', 'w') as fw:
            fw.write('\n'.join(answers))

    def adjust_test_data(self, test_path):
        """
        adjust source date, including remove unused words and do segmentation again
        :param test_path: path of test data
        :return: contents, questions
        """

        # read content
        train_text = open(test_path).read().strip()
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

        with open(TMP_RESULT + '/test.txt', 'w') as fw:
            fw.write('\n')
            for index, content in enumerate(contents):
                fw.write('text_' + str(index) + ':' + ' '.join(questions[index]) + '\n')
                for sentence in content:
                    fw.write(' '.join(sentence) + '\n')
                fw.write('\n')

        with open(TMP_RESULT + '/test_ques.txt', 'w') as fw:
            for question in questions:
                fw.write(' '.join(question) + '\n')

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

    def read_test_data(self, test_path, question_path):
        """
        read adjust data, includign source text, questions, answers
        :param test_path: file path of train_data
        :param question_path: file path of questions
        :return: content, questions, answers
        """

        train_text = open(test_path).read().strip()
        contents = re.split('\ntext_.+?\n', train_text)
        contents = [[sent.split(' ') for sent in content.strip().split('\n')] for content in contents]
        contents[0] = contents[0][1:]

        questions = open(question_path).read().strip().split('\n')
        questions = [item.split(' ') for item in questions]

        return contents, questions

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
            if re.search('谁|谁家', tmp_ques):
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

    def get_ans_sents(self, contents, questions, corpus):
        """
        get the most similar sentences all questions
        :param contents: all texts/stories
        :param questions: all questions
        :return: all extracted sentences
        """

        max_sents = []
        cnt = [0 for _ in range(7)]
        fw = open(TMP_RESULT + '/failed_matched_sents_%s.txt'% corpus, 'w')
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

        if DEBUG:print(sum(cnt[:3]), cnt)
        with open(TMP_RESULT + '/ques_matched_sents_%s.txt' % corpus, 'w') as fw:
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
                result.append((A[x - 1], y-1, x-1))
                x -= 1
                y -= 1
        return result[::-1]

    def func_for_best(self, ques, sent, sent_rel, words, included_rel, replace_pos=-1):
        """
        to find the best answer word for the question
        :param ques: word sequence of question
        :param sent: word sequence of candidate sentence
        :param sent_rel: dependency syntax analysis result for candidate sentence
        :param words: candidate phrases
        :param included_rel: admitted relation for phrases
        :param replace_pos: index of the interrogative phrase
        :return: the best phrase for the question
        """

        min_word, min_dist, min_num = None, int(0x3f3f3f3f), 0
        for word in words:
            index = sent.index(word)
            if sent_rel[index][1] not in included_rel:
                continue

            tmp_ques = ques[:]
            tmp_ques[replace_pos] = word
            res_2 = self.imp_LCS(tmp_ques, sent)

            indexes = [pos for _, pos, _ in res_2]
            index_sum_2 = 0
            for i in range(len(indexes)):
                for j in range(i + 1, len(res_2)):
                    index_sum_2 += (indexes[j] - indexes[i])
            if len(res_2) > min_num:
                min_num = len(res_2)
                min_dist = index_sum_2
                min_word = word
            if index_sum_2 < min_dist:
                min_dist = index_sum_2
                min_word = word
        return [min_word, ] if min_word else []

    def ans_for_left_WHO(self, ques, sent, sent_tag, sent_rel, words, verb_index):
        for j in range(len(sent_tag)):
            pos, protype = sent_rel[j]
            if protype == 'SBV' and pos == verb_index + 1:
                target = sent[j]
                for word in words:
                    if target in word or word in target:
                        return [target, ]

        included_rel = ['SBV']
        return self.func_for_best(ques, sent, sent_rel, words, included_rel, 0)

    def ans_for_right_WHO(self, ques, sent, sent_tag, sent_rel, words, verb_index):
        for j in range(len(sent_tag)):
            pos, protype = sent_rel[j]
            if protype == 'VOB' and pos == verb_index + 1:
                target = sent[j]
                for word in words:
                    if target in word or word in target:
                        return [target, ]

        if ques[-2:] == ['是', '小明']:
            prefix = ques[-3]
            for word in words:
                for i in range(len(sent)):
                    if i>1 and sent[i-1] == prefix and sent[i] == word:
                        return [word,]
                    if i+1<len(sent) and sent[i+1] == prefix and sent[i] == word:
                        return [word, ]

        included_rel = ['VOB']
        return self.func_for_best(ques, sent, sent_rel, words, included_rel, -1)

    def ans_for_middle_WHO(self, ques, sent, sent_tag, sent_rel, words):
        included_rel = ['SBV', 'VOB']
        replace_pos = ques.index('小明')
        return self.func_for_best(ques, sent, sent_rel, words, included_rel, replace_pos)

    def extract_ans_for_WHO(self, questions, categories, extracted_sents):
        """
        extract answers for 'WHO' questions
        :param questions: all questions
        :param categories: category for each question
        :param extracted_sents: candicated sentences for questions
        :return: None
        """

        cnt, enter_cnt, true_cnt = 0, 0, 0
        for i in range(len(questions)):
            if categories[i] != 'WHO':
                continue

            cnt += 1
            ques = [word for word in questions[i] if word not in stop_words]
            sent = [word for word in extracted_sents[i][0] if word not in stop_words]

            if '谁' in ques:
                pos = ques.index('谁')
                ques[pos] = '小明'
            elif '谁家' in ques:
                pos = ques.index('谁家')
                ques[pos:pos+1] = ['小明', '家']
            else:
                continue

            ques_tag = list(postagger.postag(ques))
            ques_rel = [(arc.head, arc.relation) for arc in list(parser.parse(ques, ques_tag))]
            sent_tag = list(postagger.postag(sent))
            sent_rel = [(arc.head, arc.relation) for arc in list(parser.parse(sent, sent_tag))]
            ques_noun = [ques[j] for j in range(len(ques)) if ques_tag[j] in ['n', 'nh', 'ns', 'nz']]
            lcs = self.imp_LCS(ques, sent)

            if pos == 0: # start WHO
                verb_index = -1
                for j in range(len(lcs)):
                    if ques_tag[lcs[j][2]] == 'v':
                        verb_index = lcs[j][1]
                        break

                bound = len(sent) if verb_index == -1 else verb_index
                tmp_words = [sent[j] for j in range(bound) if sent_tag[j] in ['n', 'nh', 'ns', 'nz']]
                tmp_words = [word for word in tmp_words if word not in ques_noun]
                tmp_words = sorted(list(set(tmp_words)))
                if len(tmp_words) > 1:
                    tmp_words = self.ans_for_left_WHO(ques, sent, sent_tag, sent_rel, tmp_words, verb_index)
                if not tmp_words: continue
                enter_cnt += 1
                ans = tmp_words[0]
                pred_answers[i] = ans
                if DEBUG:
                    if ans in answers[i] or answers[i] in ans:
                        true_cnt += 1
            elif pos == len(ques)-1: # end WHO
                verb_index = -1
                for j in range(len(lcs)-1, -1, -1):
                    if ques_tag[lcs[j][2]] == 'v' and lcs[j][0] != '是':
                        verb_index = lcs[j][1]
                        break

                bound = 0 if verb_index == -1 else verb_index
                tmp_words = [sent[j] for j in range(bound+1, len(sent)) if 'n' in sent_tag[j]]
                tmp_words = [word for word in tmp_words if word not in ques_noun]
                tmp_words = sorted(list(set(tmp_words)))
                if len(tmp_words) > 1:
                    tmp_words = self.ans_for_right_WHO(ques, sent, sent_tag, sent_rel, tmp_words, verb_index)
                if not tmp_words: continue
                enter_cnt += 1
                ans = tmp_words[0]
                pred_answers[i] = ans
                if DEBUG:
                    if ans in answers[i] or answers[i] in ans:
                        true_cnt += 1
            else: # middle WHO
                lindex, rindex = 0, 0
                for item in lcs:
                    if ques.index(item[0]) < pos:
                        lindex = item[1]
                    elif ques.index(item[0]) > pos:
                        rindex = item[1]
                        break

                tmp_ques = ques
                for j in range(pos, -1,-1):
                    if ques_rel[j][1] == 'WP':
                        tmp_ques = ques[j+1:]

                tmp_words = [sent[j] for j in range(lindex+1, rindex) if 'n' in sent_tag[j]]
                tmp_words = [word for word in tmp_words if word not in ques_noun]
                tmp_words = sorted(list(set(tmp_words)))
                if len(tmp_words) > 1:
                    tmp_words = self.ans_for_middle_WHO(tmp_ques, sent, sent_tag, sent_rel, tmp_words)
                if not tmp_words: continue
                enter_cnt += 1
                ans = tmp_words[0]
                pred_answers[i] = ans
                if DEBUG:
                    if ans in answers[i] or answers[i] in ans:
                        true_cnt += 1
        print(cnt, enter_cnt, true_cnt)

    def ans_for_VOB_WHAT(self, ques, ques_tag, ques_rel, sent, sent_tag, sent_rel):
        lcs = self.imp_LCS(ques, sent)
        verb_index = -1
        for j in range(len(lcs)):
            if ques_tag[lcs[j][2]] == 'v':
                verb_index = lcs[j][1]
                break

        bound = verb_index if verb_index else -1
        ques_noun = [ques[i] for i in range(len(ques)) if ques_tag[i] in ['n', 'ns', 'nh', 'nt', 'nz']]
        tmp_words = [sent[i] for i in range(bound+1, len(sent)) if sent_tag[i] in ['n', 'ns', 'nh', 'nt', 'nz']]
        tmp_words = [word for word in tmp_words if word not in ques_noun]
        tmp_words = sorted(set(tmp_words))
        if len(tmp_words) <= 1:
            return tmp_words

        pos = ques.index('物体')
        verb_word = ques[ques_rel[pos][0]-1]
        for i in range(len(sent)):
            if sent_rel[i][1] == 'VOB' and sent[i] in tmp_words:
                verb_index = sent_rel[i][0]-1
                sent_verb = sent[verb_index]
                if sent_verb in verb_word or verb_word in sent_verb:
                    return [sent[i], ]

        included_rel = ['VOB', 'POB']
        return self.func_for_best(ques, sent, sent_rel, tmp_words, included_rel, pos)

    def ans_for_POB_WHAT(self, ques, ques_tag, ques_rel, sent, sent_tag, sent_rel):
        ques_noun = [ques[i] for i in range(len(ques)) if 'n' in ques_tag[i]]
        tmp_words = [sent[i] for i in range(len(sent)) if 'n' in sent_tag[i]]
        tmp_words = [word for word in tmp_words if word not in ques_noun]
        tmp_words = sorted(set(tmp_words))
        if len(tmp_words) <= 1:
            return tmp_words

        pos = ques.index('物体')
        ADV_word = ques[ques_rel[pos][0] - 1]
        for i in range(len(sent)):
            if sent_rel[i][1] == 'POB' and sent[i] in tmp_words:
                ADV_index = sent_rel[i][0] - 1
                sent_ADV = sent[ADV_index]
                if sent_ADV == ADV_word:
                    return [sent[i], ]
        return []

    def ans_for_SBV_WHAT(self, ques, ques_tag, ques_rel, sent, sent_tag, sent_rel):
        ques_noun = [ques[i] for i in range(len(ques)) if 'n' in ques_tag[i]]
        tmp_words = [sent[i] for i in range(len(sent)) if 'n' in sent_tag[i]]
        tmp_words = [word for word in tmp_words if word not in ques_noun]
        tmp_words = sorted(set(tmp_words))
        if len(tmp_words) <= 1:
            return tmp_words

        included_rel = ['SBV']
        replace_pos = ques.index('物体')
        return self.func_for_best(ques, sent, sent_rel, tmp_words, included_rel, replace_pos)

    def ans_for_ATT_WHAT(self, ques, ques_tag, ques_rel, sent, sent_tag, sent_rel):
        ques_noun = [ques[i] for i in range(len(ques)) if 'n' in ques_tag[i]]
        tmp_words = [sent[i] for i in range(len(sent)) if 'n' in sent_tag[i]]
        tmp_words = [word for word in tmp_words if word not in ques_noun]
        tmp_words = sorted(set(tmp_words))
        if len(tmp_words) <= 1:
            return tmp_words

        pos = ques.index('物体')
        nex_word = ques[pos+1]
        if nex_word == '颜色':
            nex_word = '色'
        elif nex_word == '国家':
            nex_word = '国'
        for word in tmp_words:
            if nex_word in word and len(word) > len(nex_word):
                return [word, ]
        for i in range(len(sent)):
            if sent[i] in tmp_words:
                if i+1<len(sent) and sent[i+1] == nex_word:
                    return [sent[i], ]

        included_rel = ['ATT', 'VOB', 'FOB']
        replace_pos = ques.index('物体')
        return self.func_for_best(ques, sent, sent_rel, tmp_words, included_rel, replace_pos)

    def ans_for_COO_WHAT(self, ques, ques_tag, ques_rel, sent, sent_tag, sent_rel):
        pos = ques.index('物体')
        coo_word = ques[ques_rel[pos][0]]
        for i in range(len(sent)):
            if sent_rel[i][1] == 'COO':
                if coo_word == sent[sent_rel[i][0]]:
                    return [sent[i], ]
                elif sent[i] == coo_word:
                    return [sent[sent_rel[i][0]],]
        return []

    def ans_for_VERB_WHAT(self, ques, ques_tag, ques_rel, sent, sent_tag, sent_rel):
        if '干嘛' in ques:
            pos = ques.index('干嘛')
        elif '干什么' in ques:
            pos = ques.index('干什么')
        else:
            pos = None

        if pos:
            lcs = self.imp_LCS(ques, sent)
            index = lcs[-1][1]
            for i in range(index+1, len(sent)):
                if sent_tag[i] == 'v' and sent[i] not in ques:
                    return [sent[i],]

        if '为什么' in ques:
            for i in range(len(sent)):
                if i+1 < len(sent) and sent[i] == '为' and sent_rel[i][1] == 'VOB' and sent[i+1] not in ques:
                    return [sent[i+1], ]
        return []

    def extract_ans_for_WHAT(self, questions, categories, extracted_sents):
        """
        extract answers for 'WHAT' questions
        :param questions: all questions
        :param categories: category for each question
        :param extracted_sents: candicated sentences for questions
        :return: None
        """

        cnt, enter_cnt, true_cnt = 0, 0, 0
        for i in range(len(questions)):
            if categories[i] != 'WHAT':
                continue

            cnt += 1
            ques = [word for word in questions[i] if word not in stop_words]
            sent = [word for word in extracted_sents[i][0] if word not in stop_words]

            if '什么' in ques:
                pos = ques.index('什么')
                ques[pos] = '物体'

            ques_tag = list(postagger.postag(ques))
            ques_rel = [(arc.head, arc.relation) for arc in list(parser.parse(ques, ques_tag))]
            sent_tag = list(postagger.postag(sent))
            sent_rel = [(arc.head, arc.relation) for arc in list(parser.parse(sent, sent_tag))]

            if '物体' not in ques:
                tmp_words = self.ans_for_VERB_WHAT(ques, ques_tag, ques_rel, sent, sent_tag, sent_rel)
                if tmp_words:
                    enter_cnt+=1
                    ans = tmp_words[0]
                    pred_answers[i] = ans
                    if DEBUG:
                        if ans in answers[i] or answers[i] in ans:
                            true_cnt += 1
                continue

            if ques_rel[pos][1] in ['VOB']:
                tmp_words = self.ans_for_VOB_WHAT(ques, ques_tag, ques_rel, sent, sent_tag, sent_rel)
            elif ques_rel[pos][1] in ['POB']:
                tmp_words = self.ans_for_POB_WHAT(ques, ques_tag, ques_rel, sent, sent_tag, sent_rel)
            elif ques_rel[pos][1] in ['SBV']:
                tmp_words = self.ans_for_SBV_WHAT(ques, ques_tag, ques_rel, sent, sent_tag, sent_rel)
            elif ques_rel[pos][1] in ['ATT']:
                tmp_words = self.ans_for_ATT_WHAT(ques, ques_tag, ques_rel, sent, sent_tag, sent_rel)
            elif ques_rel[pos][1] in ['COO']:
                tmp_words = self.ans_for_COO_WHAT(ques, ques_tag, ques_rel, sent, sent_tag, sent_rel)
            else:
                tmp_words = []

            if not tmp_words: continue
            enter_cnt += 1
            ans = tmp_words[0]
            pred_answers[i] = ans
            if DEBUG:
                if ans in answers[i] or answers[i] in ans:
                    true_cnt += 1

        print(cnt, enter_cnt, true_cnt)

    def extract_ans_for_WHERE(self, questions, categories, extracted_sents):
        """
        extract answers for 'WHERE' questions
        :param questions: all questions
        :param categories: category for each question
        :param extracted_sents: candicated sentences for questions
        :return: None
        """

        cnt, enter_cnt, true_cnt = 0, 0, 0
        for i in range(len(questions)):
            if categories[i] != 'WHERE':
                continue

            cnt += 1
            res = self.handle_ques_sent(questions, extracted_sents, i)
            ques, ques_tag, ques_rel, sent, sent_tag, sent_rel = res

            lcs = self.imp_LCS(ques, sent)
            if '哪儿' in ques:
                pos = ques.index('哪儿')
            elif '哪里' in ques:
                pos = ques.index('哪里')
            elif '哪' in ques:
                pos = ques.index('哪')
            else:
                pos = None
            if not pos: continue

            lindex, rindex = 0, len(sent)
            for item in lcs:
                if lindex < item[2] < pos and ques_tag[item[2]] == 'v':
                    lindex = item[1] + 1
                elif pos < item[2] < rindex and ques_tag[item[2]] == 'v':
                    rindex = item[1]
            if rindex <= lindex: continue

            ques_noun = [ques[i] for i in range(len(ques)) if 'n' in ques_tag[i]]
            tmp_words = [sent[i] for i in range(lindex, rindex) if 'n' in sent_tag[i]]
            tmp_words = [word for word in tmp_words if word not in ques_noun]
            tmp_words = sorted(set(tmp_words))
            if len(tmp_words) > 1:
                included_rel = ['SBV', 'VOB', 'POB']
                tmp_words = self.func_for_best(ques, sent, sent_rel, tmp_words, included_rel, pos)

            if not tmp_words: continue
            enter_cnt += 1
            ans = tmp_words[0]
            pred_answers[i] = ans
            if DEBUG:
                if ans in answers[i] or answers[i] in ans:
                    true_cnt += 1
        print(cnt, enter_cnt, true_cnt)

    def extract_ans_for_WHICH(self, questions, categories, extracted_sents):
        """
        extract answers for 'WHICH' questions
        :param questions: all questions
        :param categories: category for each question
        :param extracted_sents: candicated sentences for questions
        :return: None
        """

        cnt, enter_cnt, true_cnt = 0, 0, 0
        for i in range(len(questions)):
            if categories[i] != 'WHICH':
                continue

            cnt += 1
            res = self.handle_ques_sent(questions, extracted_sents, i)
            ques, ques_tag, ques_rel, sent, sent_tag, sent_rel = res
            ques_noun = [ques[i] for i in range(len(ques)) if 'n' in ques_tag[i]]
            tmp_words = [sent[i] for i in range(len(sent)) if 'n' in sent_tag[i]]
            tmp_words = [word for word in tmp_words if word not in ques_noun]
            tmp_words = sorted(set(tmp_words))
            if len(tmp_words) > 1:
                pos = None
                if '哪个' in ques:
                    pos = ques.index('哪个') + 1
                elif '哪种' in ques:
                    pos = ques.index('哪种') + 1
                elif '哪' in ques:
                    pos = ques.index('哪') + 2
                if not pos or pos >= len(ques):
                    continue

                tar_word = ques[pos]
                for word in sent:
                    if word in tmp_words and word[-1] in tar_word:
                        tmp_words = [word, ]
                if len(tmp_words) > 1: tmp_words = []
            if not tmp_words: continue
            enter_cnt += 1
            ans = tmp_words[0]
            pred_answers[i] = ans
            if DEBUG:
                if ans in answers[i] or answers[i] in ans:
                    true_cnt += 1
        print(cnt, enter_cnt, true_cnt)

    def handle_ques_sent(self, questions, extracted_sents, i):
        """
        do POS and dependency syntax analysis for question and candidate sentence
        :param questions: all questions
        :param extracted_sents: candicated sentences for questions
        :param i: index of the question
        :return: None
        """

        ques = [word for word in questions[i] if word not in stop_words]
        sent = [word for word in extracted_sents[i][0] if word not in stop_words]
        ques_tag = list(postagger.postag(ques))
        ques_rel = [(arc.head, arc.relation) for arc in list(parser.parse(ques, ques_tag))]
        sent_tag = list(postagger.postag(sent))
        sent_rel = [(arc.head, arc.relation) for arc in list(parser.parse(sent, sent_tag))]
        return ques, ques_tag, ques_rel, sent, sent_tag, sent_rel

    def extract_ans_for_extra(self, questions, extracted_sents):
        """
        extract answers for the rest questions
        :param questions: all questions
        :param extracted_sents: candicated sentences for questions
        :return:　None
        """

        cnt, enter_cnt, true_cnt = 0, 0, 0
        for i in range(len(questions)):
            if pred_answers[i] != '无':
                continue

            cnt += 1
            res = self.handle_ques_sent(questions, extracted_sents, i)
            ques, ques_tag, ques_rel, sent, sent_tag, sent_rel = res

            ques_noun = [ques[j] for j in range(len(ques)) if ques_tag[j] in ['n', 'ns', 'nh', 'nt']]
            tmp_words = [sent[j] for j in range(len(sent)) if sent_tag[j] in ['n', 'ns', 'nh', 'nt']]
            tmp_words = [word for word in tmp_words if word not in ques_noun]
            tmp_words = sorted(set(tmp_words))
            if not tmp_words:
                ques_noun = [ques[j] for j in range(len(ques)) if ques_tag[j] in ['a', 'v', 'i', 'nz']]
                tmp_words = [sent[j] for j in range(len(sent)) if sent_tag[j] in ['a', 'v', 'i', 'nz']]
                tmp_words = [word for word in tmp_words if word not in ques_noun]
                tmp_words = sorted(set(tmp_words))
                if not tmp_words:
                    tmp_words = [word for word in sent if word not in ques_noun]

            included_rel = ['SBV', 'ATT', 'COO', 'VOB', 'POB', 'FOB', 'IOB']
            replaced_pos = -1
            for index, phrase in enumerate(ques):
                if re.search('谁|什么|哪|怎', phrase):
                    replaced_pos = index
            ans = self.func_for_best(ques, sent, sent_rel, tmp_words, included_rel, replaced_pos)
            if not ans:
                pred_answers[i] = tmp_words[0] if tmp_words else '无'
                continue
            pred_answers[i] = ans[0]
            ans = ans[0]
            enter_cnt += 1
            if DEBUG:
                if ans in answers[i] or answers[i] in ans:
                    true_cnt += 1
        print(cnt, enter_cnt, true_cnt)

def run_train():
    handler = RC()
    # handler.adjust_source_data('train.doc_query', 'reference.answer')
    contents, questions, answers = handler.read_adjust_data(TMP_RESULT + '/train.txt', TMP_RESULT + '/train_ques.txt', TMP_RESULT + '/answers.txt')
    # extracted_sents = handler.get_ans_sents(contents, questions, 'train')

    global pred_answers
    pred_answers = ['无' for _ in range(len(questions))]
    questions, extracted_sents = handler.read_extract_sents(TMP_RESULT + '/ques_matched_sents_train.txt')
    categories, cate_ques = handler.classify_questions(questions)

    handler.extract_ans_for_WHO(questions, categories, extracted_sents)
    handler.extract_ans_for_WHAT(questions, categories, extracted_sents)
    handler.extract_ans_for_WHERE(questions, categories, extracted_sents)
    handler.extract_ans_for_WHICH(questions, categories, extracted_sents)
    tmp_answers = pred_answers[:]
    handler.extract_ans_for_extra(questions, extracted_sents)

    oth_answers = open('optimize_simplified.txt').read().strip().split('\n')
    oth_answers = [item.split()[-1] for item in oth_answers]
    for i in range(len(pred_answers)):
        if tmp_answers[i] == '无' and 1 <= len(oth_answers[i]) <= 3:
            pred_answers[i] = oth_answers[i]

    with open(TMP_RESULT + '/pred_answer_train.txt', 'w') as fw:
        for i in range(len(questions)):
            fw.write('<qid_%d> ||| %s\n' % (i, pred_answers[i]))

def run_test():
    handler = RC()
    # handler.adjust_test_data('test.doc_query')
    contents, questions = handler.read_test_data(TMP_RESULT + '/test.txt', TMP_RESULT + '/test_ques.txt')
    # extracted_sents = handler.get_ans_sents(contents, questions, 'test')

    global pred_answers
    pred_answers = ['无' for _ in range(len(questions))]
    questions, extracted_sents = handler.read_extract_sents(TMP_RESULT + '/ques_matched_sents_test.txt')
    categories, cate_ques = handler.classify_questions(questions)

    handler.extract_ans_for_WHO(questions, categories, extracted_sents)
    handler.extract_ans_for_WHAT(questions, categories, extracted_sents)
    handler.extract_ans_for_WHERE(questions, categories, extracted_sents)
    handler.extract_ans_for_WHICH(questions, categories, extracted_sents)
    tmp_answers = pred_answers[:]
    handler.extract_ans_for_extra(questions, extracted_sents)

    oth_answers = open('Levein_answer.txt').read().strip().split('\n')
    oth_answers = [item.split()[-1] for item in oth_answers]
    for i in range(len(pred_answers)):
        if tmp_answers[i] == '无' and 1 <= len(oth_answers[i]) <= 3:
            pred_answers[i] = oth_answers[i]

    with open(TMP_RESULT + '/pred_answer_test.txt', 'w') as fw:
        for i in range(len(questions)):
            fw.write('<qid_%d> ||| %s\n' % (i, pred_answers[i]))


if __name__ == '__main__':
    if not os.path.exists(TMP_RESULT):
        os.mkdir(TMP_RESULT)

    parser = Parser()
    segmentor = Segmentor()
    postagger = Postagger()
    parser.load(par_model_path)
    segmentor.load_with_lexicon(cws_model_path, 'outer_dict.txt')
    postagger.load(pos_model_path)

    with open('stop_words.txt') as fr:
        stop_words = fr.read().strip().split('\n')

    run_train()
    print('-'*10)
    run_test()

    parser.release()
    segmentor.release()
    postagger.release()

"""
    def get_train_corpus(self, questions, extracted_sents):
        X, Y = [], []
        for i in range(len(questions)):
            ans = pred_answers[i]
            if ans not in answers[i] and  answers[i] not in ans:
                continue

            res = self.handle_ques_sent(questions, extracted_sents, i)
            ques, ques_tag, ques_rel, sent, sent_tag, sent_rel = res
            ques_noun = [ques[j] for j in range(len(ques)) if ques_tag[j] in ['n', 'ns', 'nh', 'nt', 'a', 'v', 'i', 'nz']]
            tmp_words = [(sent[j], sent_tag[j], sent_rel[j][1]) for j in range(len(sent)) if sent_tag[j] in ['n', 'ns', 'nh', 'nt', 'a', 'v', 'i', 'nz']]
            tmp_words = [item for item in tmp_words if item[0] not in ques_noun]

            ques_word_tag_rel = ['%s,%s' % (tag, rel[1]) for word, tag, rel in zip(ques, ques_tag, ques_rel)]
            sent_word_tag_rel = ['%s,%s' % (tag, rel[1]) for word, tag, rel in zip(sent, sent_tag, sent_rel)]
            prefix  = '|'.join(ques_word_tag_rel) + '||' + '|'.join(sent_word_tag_rel) + '||'
            for item in tmp_words:
                one_item = prefix + ','.join(item[1:])
                X.append(one_item)
                if item[0] in answers[i] or answers[i] in item[0]:
                    Y.append(1)
                else:
                    Y.append(0)
        with open('train_corpus.txt', 'w') as fw:
            for x, y in zip(X, Y):
                fw.write('%s\t%s' % (y, x) + '\n')

    def get_test_corpus(self, questions, extracted_sents):
        X, Y, ANS = [], [], []
        for i in range(len(questions)):
            if pred_answers[i] != '无':
                continue

            res = self.handle_ques_sent(questions, extracted_sents, i)
            ques, ques_tag, ques_rel, sent, sent_tag, sent_rel = res
            ques_noun = [ques[j] for j in range(len(ques)) if ques_tag[j] in ['n', 'ns', 'nh', 'nt', 'a', 'v', 'i', 'nz']]
            tmp_words = [(sent[j], sent_tag[j], sent_rel[j][1]) for j in range(len(sent)) if sent_tag[j] in ['n', 'ns', 'nh', 'nt', 'a', 'v', 'i', 'nz']]
            tmp_words = [item for item in tmp_words if item[0] not in ques_noun]

            ques_word_tag_rel = ['%s,%s' % (tag, rel[1]) for word, tag, rel in zip(ques, ques_tag, ques_rel)]
            sent_word_tag_rel = ['%s,%s' % (tag, rel[1]) for word, tag, rel in zip(sent, sent_tag, sent_rel)]
            prefix  = '|'.join(ques_word_tag_rel) + '||' + '|'.join(sent_word_tag_rel) + '||'
            for item in tmp_words:
                one_item = prefix + ','.join(item[1:])
                X.append(one_item)
                if item[0] in answers[i] or answers[i] in item[0]:
                    Y.append(1)
                else:
                    Y.append(0)
                ANS.append('%s\t%s' % (item[0], answers[i]))
        with open('test_corpus.txt', 'w') as fw:
            for x, y in zip(X, Y):
                fw.write('%s\t%s' % (y, x) + '\n')
        with open('ans_words.txt', 'w') as fw:
            fw.write('\n'.join(ANS))

    def train_model(self):
        with open('train_corpus.txt') as fr:
            lines = [item.split('\t') for item in fr.read().strip().split('\n')]
            train_X = [x for _, x in lines]
            train_Y = [y for y, _ in lines]
        with open('test_corpus.txt') as fr:
            lines = [item.split('\t') for item in fr.read().strip().split('\n')]
            test_X = [x for _, x in lines]
            test_Y = [y for y, _ in lines]

        # model = SVC(kernel='sigmoid')
        model = NuSVC(kernel='sigmoid')
        model.fit(train_X, train_Y, )
        # res = model.predict(test_X)
        score = model.score(test_X, test_Y)
        print(score)
"""
