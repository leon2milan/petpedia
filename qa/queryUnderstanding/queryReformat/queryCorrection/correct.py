import gc
import math
import os
import re
import time
from collections import Counter

import kenlm
import pandas as pd
import pycorrector
from config import get_cfg
from qa.matching.lexical.lexical_similarity import LexicalSimilarity
from qa.matching.lexical.tone_shape_code import TSC, VatiantKMP
from qa.queryUnderstanding.queryReformat.queryCorrection.pinyin import Pinyin
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.queryUnderstanding.representation.ngram import BiGram
from qa.tools import flatten, setup_logger
from qa.tools.bk_tree import BKTree
from qa.tools.trie import Trie

logger = setup_logger(level='info', name='correction')


def gen_ranges(seg_list):
    st = 0
    res = []
    for i in seg_list:
        res.append((st, st + len(i)))
        st += len(i)
    return res


class SpellCorrection(object):
    def __init__(self, cfg):
        logger.info('Initializing Correction Module ....')
        self.cfg = cfg
        self.init()

    def init(self):
        self.py = Pinyin(self.cfg)
        self.bigram = BiGram(self.cfg)
        self.tsc = TSC(self.cfg)
        try:
            self.bigram.load()
        except:
            text = [x.strip() for x in open(self.cfg.BASE.ROUGH_WORD_FILE).readlines()]
            bigram = BiGram(cfg)
            bigram.build(text)
            del text, bigram
            gc.collect()
            self.bigram.load()
        self.seg = Segmentation(self.cfg)
        kenlm_path = os.path.join(
            self.cfg.REPRESENTATION.KENLM.SAVE_PATH,
            '%s.arpa' % self.cfg.REPRESENTATION.KENLM.PROJECT)
        if not os.path.exists(kenlm_path):
            from qa.queryUnderstanding.representation.kenlm import KenLM
            km = KenLM(self.cfg)
            km.write_corpus(
                km.text_generator(self.cfg.BASE.CHAR_FILE, cut=False),
                km.corpus_file)  # 将语料转存为文本
            # NLM模型训练
            km.lm_train()
            # NLM模型arpa文件转化
            km.convert_format()
            km.count_ngrams()
        self.quad_model = kenlm.Model(kenlm_path)
        # self.dats = self.load_dats(
        # )  # TODO:  替代unigram 降低信息冗余, ternary tree/ trie + csr 优化bigram
        self.same_pinyin = Words(self.cfg).get_samepinyin
        self.same_stroke = Words(self.cfg).get_samestroke
        self.specialization = Words(self.cfg).get_specializewords
        try:
            self.bk = BKTree(self.cfg)
        except:
            text = [
                "".join(x.strip().split())
                for x in open(cfg.BASE.ROUGH_WORD_FILE).readlines()
            ]
            bktree = BKTree(self.cfg)
            bktree.builder(text)
            del text, bktree
            gc.collect()
            self.bk = BKTree(self.cfg)
        self.word_trie = self.load_Trie('word')
        pycorrector.correct('感帽')

    def load_Trie(self, name):
        text = Counter(
            flatten([
                x.strip().split()
                for x in open(self.cfg.BASE.ROUGH_WORD_FILE).readlines()
            ]))
        trie = Trie(self.cfg)
        try:
            trie.load(name)
        except:
            for word, _ in text.items():
                if name == 'word':
                    trie.add_word(word)
                elif name == 'pinyin':
                    trie.add_word(Pinyin.get_pinyin_list(word))
            trie.save(name)
        return trie

    def detect(self, text, segranges):
        err_pos = []
        for i in range(len(text)):
            item = text[i]
            before_item_ssc = self.tsc.getSSC(text[i - 1] if i else '' + item,
                                              'ALL')
            after_item_ssc = self.tsc.getSSC(
                item + text[i + 1] if i + 1 < len(text) else '', 'ALL')
            segrange = segranges[i]
            baseline = self.cfg.CORRECTION.THRESHOLD * 2
            before = self.bigram.bi_tf(
                text[i - 1] if i else self.cfg.BASE.START_TOKEN, item)
            after = self.bigram.bi_tf(
                item,
                text[i + 1] if i + 1 < len(text) else self.cfg.BASE.END_TOKEN)
            logger.debug(
                'item : {} , uni count: {}, baseline: {}, before: {}, after: {}'
                .format(item, self.bigram.uni_tf(item), baseline, before,
                        after))
            threshold = int(self.bigram.uni_tf(item) > 100
                            ) + self.bigram.uni_tf(item) // 1000 + int(
                                before > self.cfg.CORRECTION.THRESHOLD) + int(
                                    after > self.cfg.CORRECTION.THRESHOLD)
            if threshold > 1:
                # 过滤词频率过高， 保留前后bigram 低频出现
                logger.debug(f"{item} is passed!!!")
                continue
            if len(item) > 1 and self.bigram.uni_tf(item[::-1]) > baseline:
                # 如果词长度大于1， 判断反转之后词是否具有更高频率
                err_pos.append({
                    'word': item,
                    'st_pos': segrange[0],
                    'end_pos': segrange[1],
                    'err_score': 1.0,
                    'candidate': item[::-1],
                    'source': 'char_reverse'
                })
            if i and before < self.cfg.CORRECTION.THRESHOLD and self.bigram.bi_tf(
                    item, text[i - 1]) > before:
                # 与前词交换顺序
                err_pos.append({
                    'word': text[i - 1] + item,
                    'st_pos': segranges[i - 1][0],
                    'end_pos': segrange[1],
                    'err_score': 1.0,
                    'candidate': item + text[i - 1],
                    'source': 'before_reverse'
                })
            if i + 1 < len(
                    text
            ) and after < self.cfg.CORRECTION.THRESHOLD and self.bigram.bi_tf(
                    text[i + 1], item) > after:
                #与后词交换顺序
                err_pos.append({
                    'word': item + text[i + 1],
                    'st_pos': segrange[0],
                    'end_pos': segranges[i + 1][1],
                    'err_score': 1.0,
                    'candidate': text[i + 1] + item,
                    'source': 'after_reverse'
                })
            if i > 0 and self.bigram.bi_tf(text[i - 1], item) > 0:
                # 与前词 拼音trie tree 检索
                tmp = self.py.pinyin_candidate(text[i - 1] + item,
                                               1,
                                               method='uni')
                for can in tmp:
                    if self.bigram.uni_tf(can) > baseline:
                        tsc = LexicalSimilarity.jaccard(
                            "".join(self.tsc.getSSC(can, 'ALL')),
                            "".join(before_item_ssc)) > 0.6
                        if tsc > 0.6:
                            err_pos.append({
                                'word': text[i - 1] + item,
                                'st_pos': segranges[i - 1][0],
                                'end_pos': segrange[1],
                                'err_score': 2.0 + tsc,
                                'candidate': can,
                                'source': 'before_py'
                            })

            if i + 1 < len(text) and not self.bigram.bi_tf(item, text[i + 1]):
                # 与后词 拼音trie tree 检索
                tmp = self.py.pinyin_candidate(item + text[i + 1],
                                               1,
                                               method='uni')
                for can in tmp:
                    if self.bigram.uni_tf(can) > baseline:
                        tsc = LexicalSimilarity.jaccard(
                            "".join(self.tsc.getSSC(can, 'ALL')),
                            "".join(after_item_ssc))
                        if tsc > 0.6:
                            err_pos.append({
                                'word': item + text[i + 1],
                                'st_pos': segrange[0],
                                'end_pos': segranges[i + 1][1],
                                'err_score': 2.0 + tsc,
                                'candidate': can,
                                'source': 'after_py'
                            })

            if len(item) == 1:
                # 如果是单字， 取同音字 进行前后匹配判断
                for char in item:
                    res = self.same_pinyin.get(char, [])
                    # res += self.same_stroke.get(char, [])
                    for can in res:
                        # modify = item.replace(char, can)
                        if i and self.bigram.bi_tf(text[i - 1],
                                                   can) > baseline:
                            err_pos.append({
                                'word': item,
                                'st_pos': segrange[0],
                                'end_pos': segrange[1],
                                'err_score': 1.0,
                                'candidate': can,
                                'source': 'same'
                            })
                        elif i + 1 < len(text) and self.bigram.bi_tf(
                                can, text[i + 1]) > baseline:
                            err_pos.append({
                                'word': item,
                                'st_pos': segrange[0],
                                'end_pos': segrange[1],
                                'err_score': 1.0,
                                'candidate': can,
                                'source': 'same'
                            })
            # 实体词 匹配
            tmp = self.py.pinyin_candidate(item, 1, method='entity')
            for can in tmp:
                if self.bigram.uni_tf(can) > baseline:
                    err_pos.append({
                        'word': item,
                        'st_pos': segrange[0],
                        'end_pos': segrange[1],
                        'err_score': 3.0,
                        'candidate': can,
                        'source': 'entity_py'
                    })
            # 词 拼音trie tree 匹配
            if self.bigram.uni_tf(item) < baseline:
                tmp = self.py.pinyin_candidate(item, 1, method='uni')
                for can in tmp:
                    if ((self.bigram.bi_tf(text[i - 1], can)) or
                        (i + 1 < len(text) and self.bigram.bi_tf(
                            can, text[i + 1]))) and (can != item):
                        err_pos.append({
                            'word': item,
                            'st_pos': segrange[0],
                            'end_pos': segrange[1],
                            'err_score': 2.0,
                            'candidate': can,
                            'source': 'all_py'
                        })

        # bktree 检索整句
        bk_res = self.bk.find("".join(text), 2)
        for x in bk_res:
            err_pos.append({
                'word': x,
                'st_pos': 0,
                'end_pos': len(text),
                'pos': (0, len(text)),
                'candidate': x,
                'err_score': 3.0,
                'source': 'bktree'
            })
        logger.debug('err_pos: {}'.format(err_pos))
        if err_pos:
            err_pos = pd.DataFrame(err_pos).groupby(
                ['candidate', 'end_pos', 'source', 'st_pos', 'word'],
                as_index=False)['err_score'].agg('sum').to_dict('records')
        return err_pos

    def detect_helper(self, text):
        """
        分词列表长度为1， 且词频较低，直接返回
        遍历分词列表， 如果当前词与上下文的bigram 词频很低。 认为出错，返回当前词， 以及当前词的起始， 结束位置
        拼音编辑距离检测
        w2v model 检测
        """
        length = len("".join(text))
        if length <= 1:
            if self.bigram.uni_tf(text[0]) < self.cfg.CORRECTION.THRESHOLD * 2:
                return [{
                    'word': text[0],
                    'st_pos': 0,
                    'end_pos': len(text[0]),
                    'err_score': 1.0,
                    'candidate': []
                }]
            else:
                return []
        text = [self.cfg.BASE.START_TOKEN] + text + [self.cfg.BASE.END_TOKEN]
        err = []
        for i in range(1, len(text) - 1):
            candidate = []
            # uni_tf = self.bigram.uni_tf(text[i])
            current = text[i]
            if text[i] in self.specialization:
                continue
            err_score = 0.0

            back = self.bigram.bi_tf(text[i - 1], text[i])
            future = self.bigram.bi_tf(text[i], text[i + 1])
            # if uni_tf < self.cfg.CORRECTION.THRESHOLD:
            #     st = len("".join(text[1: i - 1]))
            #     ed = min(length, st + len("".join(text[max(1, i - 1):i + 1])))
            # print(i, text[i], back, future)
            st = len("".join(text[1:i]))
            ed = min(length, st + len("".join(text[i:i + 1])))
            if len(current) > 1:
                candidate = self.py.pinyin_candidate(current, 1)
                err_score += 0.8 * len(candidate)
            if back < self.cfg.CORRECTION.THRESHOLD:
                err_score += 0.6
            if future < self.cfg.CORRECTION.THRESHOLD:
                err_score += 0.4

            if err_score > 0:
                err.append({
                    'word': text[i],
                    'st_pos': st,
                    'end_pos': ed,
                    'err_score': err_score,
                    'candidate': candidate
                })
        return err

    def get_score(self, sentence):
        return self.quad_model.score(sentence, bos=False, eos=False)

    def correct(self, text):
        """
        1. pinyin匹配 
        2. 检测
        3. 同音同型替换: 针对单字
        4. pingyin 生成候选
        5. word 生成候选
        6. bktree 生成候选
        7. 排序
        """
        start = time.time()
        max_score = float("-inf")
        e_pos = (-1, -1)
        can = ""
        candidates = []
        py_detect = re.findall('[A-Za-z]', text)
        if "".join(py_detect) == text:
            candidates = self.py.pinyin_candidate(text, 0, method='uni')
            for x in candidates:
                uni_score = self.bigram.uni_tf(x)
                if uni_score > max_score:
                    max_score = uni_score
                    can = x
            return ((0, len(text)), can, max_score)
        # pinyin匹配
        candidates = self.py.pinyin_candidate(text, 0)
        if candidates:
            return ((0, len(text)), candidates, 1.0)
        text_list = self.seg.cut(text, is_rough=True)
        # text_list = list(text)
        logger.debug('text_list: {}'.format(text_list))
        logger.debug('segmentation takes: {}'.format(time.time() - start))
        segranges = gen_ranges(text_list)
        raw_score = self.get_score(" ".join(text_list))
        err_info = self.detect(text_list, segranges)
        query_ssc = self.tsc.getSSC(text, 'ALL')
        logger.debug('detect takes: {}'.format(time.time() - start))
        logger.info('err_info: {}'.format([(x['candidate'], x['source'])
                                           for x in err_info]))
        for err in err_info:
            sentence = text_list[:err['st_pos']] + [
                err['candidate']
            ] + text_list[err['end_pos']:]
            # sentence['candidate']
            # sentence = self.seg.cut(sentence, is_rough=True)

            # TODO 使用分类模型
            # lm model ppl 变化情况
            score = err['err_score']

            kmp = VatiantKMP(0.5)
            kmp.indexKMP(query_ssc, self.tsc.getSSC(err['candidate'], 'ALL'),
                         'ALL', self.tsc.strokesDictReverse)
            if kmp.startIdxRes:
                score += 2
            elif 'reverse' in err['source'] or len(py_detect) > 0:
                score *= 1
            else:
                continue
            score += self.get_score(" ".join(sentence)) - raw_score
            # 频次变化
            # score += self.bigram.uni_tf(candidate) - self.bigram.uni_tf(string)
            # 召回次数
            # score += freq
            # 编辑距离
            score += len(text) - LexicalSimilarity.levenshteinDistance(
                text, "".join(sentence))

            # jaccard 距离
            score += LexicalSimilarity.jaccard(
                "".join(Pinyin.get_pinyin_list(text)),
                "".join(Pinyin.get_pinyin_list("".join(sentence))))

            if score > max_score:
                max_score = score
                # can = candidate['predict']
                # e_pos = candidate['pos']
                can = err['candidate']
                e_pos = (err['st_pos'], err['end_pos'])
        logger.debug('rank takes: {}'.format(time.time() - start))
        # if not can:
        #     _, detail = pycorrector.correct(t)
        #     if detail:
        #         e_pos = (detail[0][2], detail[0][3])
        #         can = detail[0][1]
        #         max_score = 1.0
        return (e_pos, can, 0.0 if math.isinf(max_score) else max_score)


if __name__ == '__main__':
    cfg = get_cfg()
    sc = SpellCorrection(cfg)
    text = [
        "猫沙盆",  # 错字
        "我家猫猫精神没有",  # 乱序
        "狗狗发了",  # 少字
        "maomi",  # 拼音
        "猫咪啦肚子怎么办",
        "我家狗拉啦稀了",  # 多字
        "狗狗偶尔尿xie怎么办",
        "狗老抽出怎么办",
        '我家猫猫感帽了',
        '传然给我',
        '呕土不止',
        '一直咳数',
        '我想找哥医生',
        "哈士奇老拆家怎么办"
    ]
    tt = []
    for t in text:
        start = time.time()
        e_pos, candidate, score = sc.correct(t)
        tt.append(time.time() - start)
        print("raw {}, candidate {} vs: wrong {}, score: {}, takes time : {}".
              format(t, candidate, t[e_pos[0]:e_pos[1]], score,
                     time.time() - start))
        # start = time.time()
        # corrected_sent, detail = pycorrector.correct(t)
        # print(corrected_sent, detail)
        # print("pyccorrect takes time : {}".format(time.time() - start))
    print(sum(tt) / len(tt))  # 0.015135636696448693
