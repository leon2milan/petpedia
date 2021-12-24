import math
from collections import defaultdict
from functools import reduce
from operator import add

import pickle
import os
from scipy.stats import norm
import numpy as np
from qa.queryUnderstanding.querySegmentation import Segmentation
from config import get_cfg

__all__ = ['BiGram']


class BiGram:
    __slot__ = [
        'cfg', 'seg', 'wfreq', 'wwfreq', 'rewwfreq', 'token_size',
        'token2_size'
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.seg = Segmentation(cfg)
        self.wfreq = defaultdict(int)  # 单词词频
        self.wwfreq = defaultdict(int)  # 两个单词组合词频
        self.rewwfreq = defaultdict(int)  # 两个单词组合词频
        self.token_size = 0  # 单词总数
        self.token2_size = 0  # 二元组合个数

    def _add_eosbos(self, text):
        text = " ".join(text) if isinstance(text, list) else text
        text = [self.cfg.BASE.START_TOKEN
                ] + text.split() + [self.cfg.BASE.END_TOKEN]
        return " ".join(text)

    def build(self, data):
        for text in data:
            text = self._add_eosbos(text)
            word_list = list(text.split())
            size = len(word_list)
            for i in range(size - 1):
                ww = "".join(word_list[i:i + 2])
                self.wwfreq[ww] += 1
            for i in range(1, size):
                reww = "".join(word_list[i - 1:i + 1])
                self.rewwfreq[reww] += 1
            for word in word_list:
                self.wfreq[word] += 1
        self.token_size = reduce(add, self.wfreq.values())
        self.token2_size = reduce(add, self.wwfreq.values())
        self.mean = self.token2_size / len(self.wwfreq)
        self.var = np.var(list(self.wwfreq.values()))
        self.save()

    def uni_score(self, word):
        if word in self.wfreq:
            p = self.wfreq[word] / self.token_size
        else:
            p = 0.1 / self.token_size
        return p

    def bi_score(self, w1, w2):
        if w1 + w2 in self.wwfreq:
            p = self.wwfreq[w1 + w2] / self.wfreq[w1] * self.uni_score(w1)
        else:
            p = self.uni_score(w1) * self.uni_score(w2)
        return p

    def uni_tf(self, w):
        return self.wfreq.get(w, 0)

    def gaussian_score(self, w):
        return norm.cdf(x=self.wwfreq[w], loc=self.mean, scale=self.var)

    def bi_tf(self, w1, w2, bidirection=False):
        w = w1 + w2
        if bidirection:
            return self.rewwfreq.get(w, 0)
        else:
            return self.wwfreq.get(w, 0)

    def prob_sentence(self, sentence):
        """
        计算句子的2元联合概率和困惑度
        :param sentence: 未分词的整句
        :return: 概率，困惑度。dtype=float,float
        """
        word_list = [self.cfg.BASE.START_TOKEN
                     ] + self.seg.cut(sentence) + [self.cfg.BASE.END_TOKEN]
        size = len(word_list)
        prob = 1
        for i in range(size - 1):
            prob *= self.bi_score(word_list[i], word_list[i + 1])
        perplexity = math.pow(prob, -1.0 / (size - 1))
        return prob, perplexity

    def save(self):
        pickle.dump(
            self.wfreq,
            open(
                os.path.join(self.cfg.REPRESENTATION.NGRAM.SAVE_PATH,
                             'unigram.pkl'), 'wb'))
        pickle.dump(
            self.wwfreq,
            open(
                os.path.join(self.cfg.REPRESENTATION.NGRAM.SAVE_PATH,
                             'bigram.pkl'), 'wb'))
        pickle.dump(
            self.rewwfreq,
            open(
                os.path.join(self.cfg.REPRESENTATION.NGRAM.SAVE_PATH,
                             'rebigram.pkl'), 'wb'))

    def load(self):
        self.wfreq = pickle.load(
            open(
                os.path.join(self.cfg.REPRESENTATION.NGRAM.SAVE_PATH,
                             'unigram.pkl'), 'rb'))
        self.wwfreq = pickle.load(
            open(
                os.path.join(self.cfg.REPRESENTATION.NGRAM.SAVE_PATH,
                             'bigram.pkl'), 'rb'))
        self.rewwfreq = pickle.load(
            open(
                os.path.join(self.cfg.REPRESENTATION.NGRAM.SAVE_PATH,
                             'rebigram.pkl'), 'rb'))

        self.token_size = reduce(add, self.wfreq.values())
        self.token2_size = reduce(add, self.wwfreq.values())
        self.mean = self.token2_size / len(self.wwfreq)
        self.var = np.var(list(self.wwfreq.values()))


if __name__ == '__main__':
    cfg = get_cfg()
    seg = Segmentation(cfg)
    text = [x.strip() for x in open(cfg.BASE.ROUGH_WORD_FILE).readlines()]
    bigram = BiGram(cfg)
    bigram.build(text)

    bigram.load()
    print(bigram.uni_tf('猫'))
    print(bigram.uni_tf('沙盆'))
    print(bigram.bi_tf('猫', '沙盆'))