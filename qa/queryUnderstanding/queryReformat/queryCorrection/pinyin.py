# coding: utf-8
"""
声母：b p m f d t n l g k h j q x zh ch sh r z c s y w
单韵母:a o e i u ü
复韵母：ai ei ui ao ou iu ie ue er
特殊韵母:(er)
前鼻韵母：an en in un ün
后鼻韵母：ang eng ing ong
整体认读音节：zhi chi shi ri zi ci si yi wu yu yu yue yin yun yuan
"""
import os
import json
from functools import reduce
from collections import defaultdict, Counter
from pypinyin import lazy_pinyin as pinyin
from qa.queryUnderstanding.querySegmentation.words import Words
from qa.queryUnderstanding.preprocess.preprocess import is_other
from qa.tools import setup_logger, flatten
from qa.tools.trie import Trie
from config import get_cfg

logger = setup_logger()


class Pinyin:
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            self.entity_trie = Trie(cfg)
            self.entity_trie.load('entity_py')
            self.entity2py = json.load(
                open(
                    os.path.join(self.cfg.CORRECTION.MODEL_FILE,
                                 'entity_py_word2py.json')))
            self.py2entity = json.load(
                open(
                    os.path.join(self.cfg.CORRECTION.MODEL_FILE,
                                 'entity_py_py2word.json')))
            self.all_py_trie = Trie(cfg)
            self.all_py_trie.load('all_py')
            self.all_word2py = json.load(
                open(
                    os.path.join(self.cfg.CORRECTION.MODEL_FILE,
                                 'all_py_word2py.json')))
            self.all_py2word = json.load(
                open(
                    os.path.join(self.cfg.CORRECTION.MODEL_FILE,
                                 'all_py_py2word.json')))
        except:
            import traceback
            print(traceback.print_exc())
            self.entity2py, self.py2entity, self.entity_trie = self.build(
                flatten([
                    v for k, v in reduce(lambda a, b: dict(a, **b),
                                         Words(self.cfg).get_specializewords.values()).items()
                ]), 'entity_py')
            self.all_word2py, self.all_py2word, self.all_py_trie = self.build([
                k for k, v in Counter(
                    flatten([
                        x.strip().split() for x in open(
                            self.cfg.BASE.ROUGH_WORD_FILE).readlines()
                    ])).items() if len(k) > 1 and not is_other(k)
                and v > self.cfg.CORRECTION.THRESHOLD * 3
            ], 'all_py')

    @staticmethod
    def get_pinyin_list(text):
        """"""
        s = []
        result = pinyin(text)
        for x in result:
            # 字符串替换 翘舌音替换等...
            x = x.replace("ng",
                          "n").replace("zh",
                                       "z").replace("sh",
                                                    "s").replace("ch", "c")
            # x = x.replace("h", "f").replace("l", "n")
            s.append(x)
        return s

    def build(self, data, name):
        word2py = defaultdict(list)
        py2word = defaultdict(list)
        py_trie = Trie(self.cfg)
        for word in data:
            pinyin = "".join(Pinyin.get_pinyin_list(word))
            word2py[word].append(pinyin)
            py2word[pinyin].append(word)
            py_trie.add_word(pinyin)
        py_trie.save(name)
        with open(
                os.path.join(self.cfg.CORRECTION.MODEL_FILE,
                             name + '_word2py.json'), 'w') as f:
            json.dump(word2py, f, ensure_ascii=False)
        with open(
                os.path.join(self.cfg.CORRECTION.MODEL_FILE,
                             name + '_py2word.json'), 'w') as f:
            json.dump(py2word, f, ensure_ascii=False)
        return word2py, py2word, py_trie

    def pinyin_candidate(self, word, edit_dis=1, method='entity'):
        pinyin = "".join(Pinyin.get_pinyin_list(word))
        if method == 'entity':
            result = self.entity_trie.kDistance(pinyin, edit_dis)
            return list(set(flatten([self.py2entity[i] for i in result])))
        else:
            result = self.all_py_trie.kDistance(pinyin, edit_dis)
            return list(set(flatten([self.all_py2word[i] for i in result])))


if __name__ == "__main__":
    cfg = get_cfg()
    py = Pinyin(cfg)
    import time
    logger.debug("*" * 10 + "测试" + "*" * 10)
    start = time.time()
    print(py.pinyin_candidate('哈士骑'))
    print('taks time:', time.time() - start)
    print(py.pinyin_candidate('咳数'))
    print('taks time:', time.time() - start)
    print(py.pinyin_candidate('抽出'))
    print('taks time:', time.time() - start)
    print(py.pinyin_candidate('maomi', 0))
    print('taks time:', time.time() - start)

    print(py.pinyin_candidate('啦肚子', method='aaa'))
    print('all pinyin taks time:', time.time() - start)
    print(py.pinyin_candidate('尿xie', method='aaa'))
    print('all pinyin taks time:', time.time() - start)
    print(py.pinyin_candidate('呕土', method='aaa'))
    print('all pinyin taks time:', time.time() - start)
    print(py.pinyin_candidate('maomi', 0, method='aaa'))
    print('all pinyin taks time:', time.time() - start)
    print(py.pinyin_candidate('抽出', method='aaa'))
    print('all pinyin taks time:', time.time() - start)