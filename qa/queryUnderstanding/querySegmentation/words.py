"""
获取敏感词       get_sensitive_words
获取重要词       get_important_words
获取近义词       get_synonym_words
获取停用词
"""
from qa.tools import Singleton, setup_logger, flatten
from qa.tools.mongo import Mongo
import os
import pandas as pd
from functools import reduce
from collections import defaultdict

logger = setup_logger()
__all__ = ['Words']


@Singleton
class Words():
    """
    获取各种词库
    """

    _first_init = True
    __slot__ = [
        'cfg', 'mongo', '_first_init', 'stop_words', 'sensitive_words',
        'specialize_words', 'same_pinyin', 'same_stroke', 'synonym_words'
    ]

    def __init__(self, cfg):
        self.cfg = cfg

        self.mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)

        if self._first_init:  # 默认True 第一次初始化置为False
            self._first_init = False
            self.stop_words = None
            self.sensitive_words = None  # 敏感词
            self.specialize_words = None  #  专有词
            self.synonym_words = None  # 同义词
            self.same_pinyin = None  # 同音
            self.same_stroke = None  # 同型
            self.init()

    @staticmethod
    def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        return '\u4e00' <= uchar <= '\u9fa5'

    @staticmethod
    def is_chinese_string(string):
        """判断是否全为汉字"""
        return all(Words.is_chinese(c) for c in string)

    @staticmethod
    def is_number(uchar):
        """判断一个unicode是否是数字"""
        return u'u0030' <= uchar <= u'u0039'

    @staticmethod
    def is_number_string(string):
        """判断是否全部是数字"""
        return all(Words.is_number(c) for c in string)

    @staticmethod
    def is_alphabet(uchar):
        """判断一个unicode是否是英文字母"""
        return u'u0041' <= uchar <= u'u005a' or u'u0061' <= uchar <= u'u007a'

    @staticmethod
    def is_alphabet_string(string):
        """判断是否全部为英文字母"""
        return all('a' <= c <= 'z' for c in string)

    @staticmethod
    def is_other(uchar):
        """判断是否非汉字，数字和英文字符"""
        return not (Words.is_chinese(uchar) or Words.is_number(uchar)
                    or Words.is_alphabet(uchar))

    def is_stopword(self, s):
        return s in self.stop_words

    def is_disease(self, s):
        return s in self.__disease

    def is_symptom(self, s):
        return s in self.__symptom

    def is_dog(self, s):
        return s in self.__dog

    def is_cat(self, s):
        return s in self.__cat

    def get_alias(self, x):
        return self.__name2alias.get(x, "")

    def get_name(self, x):
        return self.__alias2name.get(x, "")

    def get_class(self, x):
        return self.__name2class.get(x, "")

    @property
    def get_stopwords(self):
        return self.stop_words

    @property
    def get_sensitivewords(self):
        return self.sensitive_words

    @property
    def get_specializewords(self):
        return self.specialize_words

    @property
    def get_samestroke(self):
        return self.same_stroke

    @property
    def get_samepinyin(self):
        return self.same_pinyin

    @property
    def get_tsc_words(self):
        return self.tsc

    def init(self):
        """相当于run() 同义词先跑"""
        self.get_synonym_words(
        )  # get_synonym_words 必须在 get_stop_words 之前, 对停用词采用同义词替换
        self.get_stop_words()
        self.get_specialize_words()
        self.get_sensitive_words()
        self.get_pinyin()
        self.get_stroke()
        self.get_tsc()
        return True

    def reload(self):
        return self.init()

    def get_synonym_words(self):
        pass

    def get_sensitive_words(self):
        """从 mongo 中获取敏感词表
        :param mongo: obj
        :return: dict {subsystem1:[word1,word2,..], subsystem2:[word11,word12...],...}
        """
        sw = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.SENSETIVE_COLLECTION,
                                 {})))[['word']]
        self.sensitive_words = sw['word'].tolist()

    def get_stop_words(self):
        self.stop_words = [
            x.strip()
            for x in open(self.cfg.DICTIONARY.STOP_WORDS).readlines()
        ]

    def get_tsc(self):
        tsc = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.TSC_COLLECTION, {})))
        res = {}

        for i in [
                "fourcorner_encode", "han_structure", "shenmu_encode",
                "stroke", "stroke_map", "structure_map", "tsc_map",
                "yunmu_encode"
        ]:
            if i == 'stroke_map':
                tmp = tsc[['stroke', i]].dropna().drop_duplicates()
                res[i] = dict(zip(tmp['stroke'], tmp[i]))
            elif i == 'structure_map':
                tmp = tsc[['han_structure', i]].dropna().drop_duplicates()
                res[i] = dict(zip(tmp['han_structure'], tmp[i]))
                res[i]['0'] = '0'
            else:
                res[i] = dict(
                    zip(tsc.iloc[tsc[i].dropna().index]['content'].dropna(),
                        tsc[i].dropna()))
        self.tsc = res

    @staticmethod
    def normalize(dic):
        return {
            k.strip(): [x.strip() for x in v.split('|') if x]
            for k, v in dic.items()
        }

    def get_specialize_words(self):
        sp = pd.DataFrame(
            list(self.mongo.find(self.cfg.BASE.SPECIALIZE_COLLECTION,
                                 {})))[['name', 'alias', 'type']]
        self.specialize_words = sp.groupby('type')['name', 'alias'].apply(
            lambda x: dict(zip(x['name'], x['alias']))).to_dict()

        self.__disease = list(self.specialize_words.get('DISEASE', {}).keys())
        self.__symptom = list(self.specialize_words.get('SYMPTOMS', {}).keys())
        self.__dog = list(self.specialize_words.get('DOG', {}).keys())
        self.__cat = list(self.specialize_words.get('CAT', {}).keys())
        self.__name2class = {
            n: c
            for c, b in self.specialize_words.items() for n, _ in b.items()
        }
        self.__name2alias = reduce(lambda a, b: dict(a, **b),
                                   self.specialize_words.values())
        tmp = [(j if j else k, k) for k, v in self.__name2alias.items()
               for j in v] + [(k, k) for k, _ in self.__name2alias.items()]
        self.__alias2name = defaultdict(list)
        for k, v in tmp:
            self.__alias2name[k].append(v)

        new_word_path = os.path.join(self.cfg.BASE.DATA_PATH,
                                     'dictionary/segmentation/new_word.csv')
        if os.path.exists(new_word_path):
            new_word = [x.strip() for x in open(new_word_path).readlines()]

        merge_all = reduce(lambda a, b: dict(a, **b),
                           self.specialize_words.values())
        tmp = list(
            set(
                list(merge_all.keys()) +
                [y for x in merge_all.values() for y in x if y]))

        custom_word = sorted([
            x.strip()
            for x in open(self.cfg.DICTIONARY.CUSTOM_WORDS).readlines()
        ])
        tmp = sorted(list(set(tmp + new_word + custom_word)))

        if len(tmp) > len(custom_word):
            with open(self.cfg.DICTIONARY.CUSTOM_WORDS, 'w') as f:
                for word in tmp:
                    f.write(word + '\n')

    def get_pinyin(self):
        self.same_pinyin = {
            line.strip().split('\t')[0]:
            list("".join(line.strip().split('\t')[1:]))
            for line in open(
                self.cfg.DICTIONARY.SAME_PINYIN_PATH).readlines()[1:]
        }

    def get_stroke(self):
        self.same_stroke = {
            line.strip().split('\t')[0]:
            list("".join(line.strip().split('\t')[1:]))
            for line in open(
                self.cfg.DICTIONARY.SAME_STROKE_PATH).readlines()[1:]
        }


if __name__ == "__main__":
    pass
