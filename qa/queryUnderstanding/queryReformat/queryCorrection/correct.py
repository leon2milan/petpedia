import collections
import os
import re
import time
from functools import partial, reduce
from itertools import chain
from typing import Counter, Iterator
import gc
import kenlm
import pycedar
import pycorrector
from config import get_cfg
from qa.matching.lexical.lexical_similarity import LexicalSimilarity
from qa.tools.bktree.BKTree import BKTree
from qa.queryUnderstanding.queryReformat.queryCorrection.pinyin import Pinyin
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.queryUnderstanding.representation.ngram import BiGram
from qa.tools import flatten, setup_logger
from qa.tools.trie import Trie

logger = setup_logger(level='info', name='correction')


def ngram(seq: str, n: int) -> Iterator[str]:
    return (seq[i:i + n] for i in range(0, len(seq) - n + 1))


def allngram(seq: str, minn=1, maxn=None) -> Iterator[str]:
    lengths = range(minn, maxn) if maxn else range(minn, len(seq))
    ngrams = map(partial(ngram, seq), lengths)
    return set(chain.from_iterable(ngrams))


def gen_ranges(seg_list):
    st = 0
    res = []
    for i in seg_list:
        res.append((st, st + len(i)))
        st += len(i)
    return res


def overlap(l1, l2):
    # Detect whether two intervals l1 and l2 overlap
    # inputs: l1, l2 are lists representing intervals
    if l1[0] < l2[0]:
        if l1[1] <= l2[0]:
            return False
        else:
            return True
    elif l1[0] == l2[0]:
        return True
    else:
        if l1[0] >= l2[1]:
            return False
        else:
            return True


def get_ranges(outranges, segranges):
    # Get the overlap ranges of outranges and segranges
    # outranges: ranges corresponding to score outliers
    # segranges: ranges corresponding to word segmentation scores
    overlap_ranges = set()
    for segrange in segranges:
        for outrange in outranges:
            if overlap(outrange, segrange):
                overlap_ranges.add(tuple(segrange))
    return [list(overlap_range) for overlap_range in overlap_ranges]


def merge_ranges(ranges):
    # Merge overlapping ranges
    # ranges: list of ranges
    ranges.sort()
    saved = ranges[0][:]
    results = []
    for st, en in ranges:
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            results.append(saved[:])
            saved[0] = st
            saved[1] = en
    results.append(saved[:])
    return results


class SpellCorrection(object):
    def __init__(self, cfg):
        logger.info('Initializing Correction Module ....')
        self.cfg = cfg
        self.init()

    def init(self):
        self.py = Pinyin(self.cfg)
        self.bigram = BiGram(self.cfg)
        try:
            self.bigram.load()
        except:
            text = [
                x.strip()
                for x in open(self.cfg.BASE.ROUGH_WORD_FILE).readlines()
            ]
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
        self.dats = self.load_dats(
        )  # TODO:  替代unigram 降低信息冗余, ternary tree/ trie + csr 优化bigram
        self.same_pinyin = Words(self.cfg).get_samepinyin
        self.same_stroke = Words(self.cfg).get_samestroke
        self.specialization = Words(self.cfg).get_specializewords
        try:
            self.bk = BKTree(self.cfg)
        except:
            text = [
                "".join(x.strip().split())
                for x in open(cfg.BASE.FINE_WORD_FILE).readlines()
            ]
            tree = BKTree(cfg)
            tree.builder(text)
            del text, tree
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

    def load_dats(self):
        trie = pycedar.dict()
        if os.path.exists(self.cfg.CORRECTION.DATS_PATH):
            trie.load(self.cfg.CORRECTION.DATS_PATH)
        else:
            text = collections.Counter(
                flatten([
                    x.strip().split()
                    for x in open(self.cfg.BASE.ROUGH_WORD_FILE).readlines()
                ]))
            trie = pycedar.dict()
            for word, cnt in text.items():
                trie[word] = cnt
            trie.save(self.cfg.CORRECTION.DATS_PATH)
        return trie

    def detect(self, text, segranges):
        err_pos = self.detect_helper(text)
        # print("assss", merge_ranges(get_ranges(err_pos, segranges)))
        # err = SpellCorrection.mergeIntervals(err)
        # raw = "".join(text)
        # err_seq = [raw[x[0]: x[1]] for x in err]
        # common = SpellCorrection.mul_string_lcs(err_seq)
        return err_pos

    @staticmethod
    def mul_string_lcs(sequences):
        if len(sequences) < 2:
            return sequences
        maxn = min(map(len, sequences))
        seqs_ngrams = map(partial(allngram, maxn=maxn), sequences)
        intersection = reduce(set.intersection, seqs_ngrams)
        if len(intersection) < 1:
            return sequences
        longest = max(intersection, key=len)
        return [longest]

    @staticmethod
    def mergeIntervals(intervals):
        if len(intervals) < 2:
            return intervals
        merged = []
        for i in range(1, len(intervals)):
            previous = intervals[i - 1]
            current = intervals[i]
            if current[0] == previous[0]:
                merged.append((current[0], max(current[1], previous[1])))
            elif current[1] == previous[1]:
                merged.append((min(current[0], previous[0]), current[1]))
            # else:
            #     merged.append(current)
        return list(set(merged))

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
        if "".join(re.findall('[A-Za-z]', text)) == text:
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
        logger.debug('text_list: {}'.format(text_list))
        logger.debug('segmentation takes: {}'.format(time.time() - start))
        segranges = gen_ranges(text_list)
        raw_score = self.get_score(" ".join(text_list))
        err_info = self.detect(text_list, segranges)
        logger.debug('detect takes: {}'.format(time.time() - start))
        logger.debug('err_info: {}'.format(err_info))
        if err_info:
            for info in err_info:
                st = info['st_pos']
                end = info['end_pos']
                err_score = info['err_score']
                string = text[st:end]
                idx = segranges.index((st, end))
                if idx and idx + 1 < len(segranges):
                    back = self.bigram.bi_tf(text_list[idx - 1],
                                             text_list[idx])
                    future = self.bigram.bi_tf(text_list[idx],
                                               text_list[idx + 1])
                else:
                    back = 0
                    future = 0

                # entity pinyin recall
                candidates += [{
                    'candidate': text[:st] + x + text[end:],
                    'pos': (st, end),
                    'predict': x,
                    'err_score': err_score
                } for x in info['candidate']]
                logger.debug('entity recall: {}'.format(info['candidate']))
                logger.debug('entity takes: {}'.format(time.time() - start))

                # same pinyin / stroke recall
                if len(string) == 1:
                    py = self.same_pinyin.get(string, [])
                    # stroke = self.same_stroke.get(string, [])
                    res = py  # + stroke
                    if res:
                        for x in res:
                            if (
                                    idx and
                                    self.bigram.bi_tf(text_list[idx - 1], x) >
                                    max(back, self.cfg.CORRECTION.THRESHOLD)
                            ) or (idx + 1 < len(segranges) and
                                  self.bigram.bi_tf(x, text_list[idx + 1]) >
                                  max(future, self.cfg.CORRECTION.THRESHOLD)):
                                candidates.append({
                                    'candidate':
                                    text[:st] + x + text[end:],
                                    'pos': (st, end),
                                    'predict':
                                    x,
                                    'err_score':
                                    err_score
                                })
                    logger.debug('same recall: {}'.format(candidates))
                    logger.debug('same takes: {}'.format(time.time() - start))
                # unigram pinyin recall
                else:
                    raw = text_list[idx]
                    res = self.py.pinyin_candidate(raw, 1, method='uni')
                    for x in res:
                        if (idx and self.bigram.bi_tf(text_list[idx - 1], x) >
                                max(back, self.cfg.CORRECTION.THRESHOLD)
                            ) or (idx + 1 < len(segranges) and
                                  self.bigram.bi_tf(x, text_list[idx + 1]) >
                                  max(future, self.cfg.CORRECTION.THRESHOLD)):
                            candidates.append({
                                'candidate':
                                text[:st] + x + text[end:],
                                'pos': (st, end),
                                'predict':
                                x,
                                'err_score':
                                err_score
                            })
                    logger.debug('unigram pinyin recall: {}'.format(res))
                    logger.debug(
                        'unigram pinyin takes: {}'.format(time.time() - start))

                # context pinyin recall
                if idx + 1 < len(segranges):
                    forward_raw = "".join((text_list[idx:idx + 2]))
                    forward = self.py.pinyin_candidate(forward_raw,
                                                       1,
                                                       method='uni')
                    logger.debug('pinyin search takes: {}'.format(time.time() -
                                                                  start))
                    forward = [
                        x for x in forward
                        if len(set(x)
                               & set(forward_raw)) >= 1 and x != forward_raw
                    ]
                    if idx and future > self.cfg.CORRECTION.THRESHOLD and back > self.cfg.CORRECTION.THRESHOLD:
                        surround_raw = "".join(text_list[idx - 1:idx + 2])
                        surround = self.py.pinyin_candidate(surround_raw,
                                                            1,
                                                            method='uni')
                        print('pinyin search 1 takes: {}'.format(time.time() -
                                                                 start))
                        surround = [
                            x for x in surround
                            if len(set(x) & set(surround_raw)) >= 1
                            and x != surround_raw
                        ]
                    else:
                        surround = []
                    for x in forward:
                        candidates.append({
                            'candidate':
                            text[:st] + x +
                            text[end + len(text_list[idx + 1]):],
                            'pos': (st, end + len(text_list[idx + 1])),
                            'predict':
                            x,
                            'err_score':
                            err_score
                        })
                    for x in surround:
                        candidates.append({
                            'candidate':
                            text[:st - len(text_list[idx - 1])] + x +
                            text[end + len(text_list[idx + 1]):],
                            'pos': (st - len(text_list[idx - 1]),
                                    end + len(text_list[idx + 1])),
                            'predict':
                            x,
                            'err_score':
                            err_score
                        })
                    logger.debug('context pinyin recall: {}'.format(forward +
                                                                    surround))
                logger.debug('context pinyin takes: {}'.format(time.time() -
                                                               start))

            bk_res = self.bk.find(string, 2)
            for x in bk_res:
                candidates.append({
                    'candidate': x,
                    'pos': (0, len(text)),
                    'predict': x,
                    'err_score': err_score
                })
            logger.debug('bktre recall: {}'.format(bk_res))
            logger.debug('bktree takes: {}'.format(time.time() - start))
            for candidate in candidates:
                sentence = candidate['candidate']
                # sentence = self.seg.cut(sentence, is_rough=True)

                # TODO 使用分类模型
                # lm model ppl 变化情况
                score = self.get_score(" ".join(sentence)) - raw_score
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
                    can = candidate['predict']
                    e_pos = candidate['pos']
            logger.debug('rank takes: {}'.format(time.time() - start))
        if not can:
            _, detail = pycorrector.correct(t)
            if detail:
                e_pos = (detail[0][2], detail[0][3])
                can = detail[0][1]
                max_score = 1.0
        return (e_pos, can, max_score)


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
        '我想找哥医生'
    ]
    for t in text:
        start = time.time()
        e_pos, candidate, score = sc.correct(t)
        print("raw {}, candidate {} vs: wrong {}, takes time : {}".format(
            t, candidate, t[e_pos[0]:e_pos[1]],
            time.time() - start))

        # start = time.time()
        # corrected_sent, detail = pycorrector.correct(t)
        # print(corrected_sent, detail)
        # print("pyccorrect takes time : {}".format(time.time() - start))
