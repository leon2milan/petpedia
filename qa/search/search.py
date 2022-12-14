import itertools
import time
import re

from config import get_cfg
from qa.queryUnderstanding.preprocess.preprocess import clean
from qa.queryUnderstanding.queryReformat.queryNormalization import \
    Normalization
from qa.retrieval.retrieval import BasicSearch
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.tools import setup_logger
from qa.tools.utils import flatten
from qa.contentUnderstanding.content_profile import ContentUnderstanding

logger = setup_logger(name='Search Advance')
__all__ = ['AdvancedSearch']


class AdvancedSearch():
    __slot__ = ['cfg', 'normalize', 'bs', 'sc', 'cs', 'seg', 'stopwords']

    def __init__(self, cfg) -> None:
        logger.info('Initializing Advanced Search Object ....')
        self.cfg = cfg
        self.normalize = Normalization(self.cfg)
        self.seg = Segmentation(cfg)
        self.stopwords = Words(cfg).get_stopwords
        self.bs = BasicSearch(self.cfg)
        if self.cfg.RETRIEVAL.DO_CORRECT:
            from qa.queryUnderstanding.queryReformat.queryCorrection.correct import \
                SpellCorrection
            self.sc = SpellCorrection(self.cfg)
        self.cs = ContentUnderstanding(self.cfg)

    def __format_to(self, result, fmt):
        if fmt == "LIST":
            l = []
            for i in result:
                l.append(i)
            return l
        return result

    def print_result(self, result):
        """
        args:
        result is a map
        [] -> BEST_MATCH ...
        [BEST_MATCH] is a list
        [BEST_MATCH][0] -> {"docid", "doc", "score"}
        """
        for i in result:
            logger.info("match_type: {}, docid: {}, score: {}.".format(
                i['match_type'], i["docid"], i["score"]))

    def __left_sentence_bound(self, pos, content):
        """
        从一个位置开始, 向左寻找句子的开头
        return
           句子开头的位置
        """
        start = pos
        stopchar = {}
        stopchar[" "] = 1
        stopchar["."] = 1
        stopchar[";"] = 1
        stopchar["\n"] = 1
        stopchar["\r"] = 1
        stopchar["。"] = 1
        stopchar["；"] = 1
        i = 0
        while start > 0:
            i += 1
            if i < 48:
                if content[start] in stopchar:
                    i -= 1
                start -= 1
                continue
            if content[start] in stopchar:
                return start + 1
            start -= 1
            if i > 300:
                if content[start] > " " and content[start] < "~":
                    break

        return start

    def __right_sentence_bound(self, pos, content):
        """
        从一个位置开始, 向右边寻找句子的结尾
        return
           句子结尾的位置
        """
        start = pos
        stopchar = {}
        stopchar[" "] = 1
        stopchar["."] = 1
        stopchar[";"] = 1
        stopchar["\n"] = 1
        stopchar["\r"] = 1
        stopchar["。"] = 1
        stopchar["；"] = 1
        i = 0
        while start < len(content):
            i += 1
            if i < 128:
                if content[start] in stopchar:
                    i -= 1
                start += 1
                continue
            if content[start] in stopchar:
                return start + 1
            start += 1
            if i > 300:
                if content[start] > " " and content[start] < "~":
                    break
        return start

    def __abstract_one_by_query(self, query, filecontent):
        fc = filecontent
        if query in fc:
            pos = fc.find(query)
            left = self.__left_sentence_bound(pos, fc)
            right = self.__right_sentence_bound(pos, fc)
            ab = fc[left:right]
            ab = ab.replace("\n", "")
            ab = ab.replace("  ", " ")
            ab = ab.replace("  ", " ")
            ab = ab.replace("\xc2\xa0", "")
            ab = ab.replace("  ", " ")
            ab = ab.replace("\r", "")
            # 飘红
            keyred = """<span style="color:#F08080">%s</span>""" % (query)
            ab = ab.replace(query, keyred)
            ab = ab.replace("__hasssss__iiiiiimage__", "")
            return ab

        tmp_query = query.replace(" ", '')
        if tmp_query in fc:
            pos = fc.find(tmp_query)
            left = self.__left_sentence_bound(pos, fc)
            right = self.__right_sentence_bound(pos, fc)
            ab = fc[left:right]
            ab = ab.replace("\n", "")
            ab = ab.replace("  ", " ")
            ab = ab.replace("  ", " ")
            ab = ab.replace("\xc2\xa0", "")
            ab = ab.replace("  ", " ")
            ab = ab.replace("\r", "")
            # 飘红
            keyred = """<span style="color:#F08080">%s</span>""" % (tmp_query)
            ab = ab.replace(tmp_query, keyred)
            ab = ab.replace("__hasssss__iiiiiimage__", "")
            return ab

        terms = query.split(' ')
        abstract_list = []
        for t in terms:
            pos = fc.find(t)
            if pos < 0:
                continue
            left = self.__left_sentence_bound(pos, fc)
            right = self.__right_sentence_bound(pos, fc)

            ab = fc[left:right]
            ab = ab.replace("\n", "")
            ab = ab.replace("  ", " ")
            ab = ab.replace("\xc2\xa0", "")
            ab = ab.replace("  ", " ")
            ab = ab.replace("  ", " ")
            ab = ab.replace("\r", "")
            keyred = """<span style="color:#F08080">%s</span>""" % (t)
            ab = ab.replace(t, keyred)
            ab = ab.replace("__hasssss__iiiiiimage__", "")
            abstract_list.append(ab)

        if len(abstract_list):
            combine = "...".join(abstract_list)
            return combine[3:]
        return ""

    def __abstract_one(self, query_list, ind):
        for q in query_list:
            query = q["query"]
            txt = self.__abstract_one_by_query(query, ind)
            if txt != "":
                return txt

    def __abstract(self, query_list, result):
        """
        result is a list
        [0] -> {"doc":, "docid":, "score":}
        """

        for j in result:
            ind = j["doc"]
            ind = ind['question']
            j["abstract"] = self.__abstract_one(query_list, ind)

    def __query_analyze_has(self, query, operations):
        if query is None or query == "":
            return
        r = operations
        # 精确不切词查询
        r.append({"query": query, "type": "BEST_MATCH"})

        # 去掉空格
        if query.replace(" ", "") != query:
            r.append({"query": query.replace(" ", ""), "type": "BEST_MATCH"})

        # 空格分割, 全排列
        if len(query.split(' ')) <= 4 and len(query.split(' ')) > 1:
            ex = list(itertools.permutations(query.split(' ')))
            for i in ex:
                r.append({"query": ' '.join(i), "type": "BEST_MATCH"})

        # WELL_MATCH 会按空格切割多个 term, 拉链归并
        if ' ' in query:
            r.append({"query": query, "type": "WELL_MATCH"})
            r.append({"query": query, "type": "PART_MATCH"})

        cutword = self.fine_lac.cut(query, cut_all=False)
        cutquery = ' '.join(cutword)
        cutquery = re.sub('\s+', ' ', cutquery)
        cutquery = cutquery.replace(".", "")
        cutquery = cutquery.encode('utf-8')

        r.append({"query": cutquery, "type": "WELL_MATCH"})
        r.append({"query": cutquery, "type": "PART_MATCH"})

        # 剔除一个空格
        while cutquery.find(' ') > 0:
            cutquery = cutquery[:cutquery.
                                find(' ')] + cutquery[cutquery.find(' ') + 1:]
            r.append({"query": cutquery, "type": "WELL_MATCH"})
            r.append({"query": cutquery, "type": "PART_MATCH"})

    def __query_analyze_not(self, operations, query):
        if query is None or query == "":
            return
        for i in operations:
            i["mustnot"] = query

    def __preprocess(self, query, _clean=True, cut=False, is_rough=True):
        query = query.strip()
        if query == "":
            return ""
        if _clean:
            query = clean(query,
                          is_tran=self.cfg.RETRIEVAL.HANDLE_TRANS,
                          has_emogi=self.cfg.RETRIEVAL.HANDLE_EMOGI)
        if cut:
            query = [
                x for x in self.seg.cut(query, is_rough=is_rough)
                if x not in self.stopwords
            ]
        return query

    def __query_analyze(self, query):
        """
            把原始 query 扩展出多个检索原语
            [ {"query": "python 第五方", "type": "BEST_MATCH"}, {}, {} ]
        """
        # query 纠错
        s = time.time()
        if self.cfg.RETRIEVAL.DO_CORRECT:
            e_pos, candidate, score = self.sc.correct(query)
            if candidate:
                query = query[:e_pos[0]] + candidate + query[e_pos[1]:]
        logger.debug('Correction takes: {}'.format(time.time() - s))

        # query 归一
        if self.cfg.RETRIEVAL.DO_NORMALIZE:
            pass
        r = []
        content_tag = self.cs.understanding(query)

        # query 扩展
        if self.cfg.RETRIEVAL.DO_EXPANSION:
            pass

        rough_query = self.__preprocess(query,
                                        _clean=False,
                                        cut=True,
                                        is_rough=True)

        fine_query = self.__preprocess(query,
                                       _clean=False,
                                       cut=True,
                                       is_rough=False)
        # BEST_MATCH： query 归一/原句 精确匹配 + knowled graph
        r.append({
            "query":
            query,
            "fine_query":
            fine_query if self.cfg.RETRIEVAL.WELL_ROUTE.SEG_GRAINED
            in ['fine', 'both'] else None,
            "rough_query":
            rough_query if self.cfg.RETRIEVAL.WELL_ROUTE.SEG_GRAINED
            in ['rough', 'both'] else None,
            "type":
            "BEST_MATCH",
            "filter": {
                'species': content_tag['SPECIES'],
                'sex': content_tag['SEX'],
                'age': content_tag['AGE'],
            }
        })

        # WELL_MATCH： query 原句  模糊搜索
        r.append({
            "query":
            query,
            "fine_query":
            fine_query if self.cfg.RETRIEVAL.WELL_ROUTE.SEG_GRAINED
            in ['fine', 'both'] else None,
            "rough_query":
            rough_query if self.cfg.RETRIEVAL.WELL_ROUTE.SEG_GRAINED
            in ['rough', 'both'] else None,
            "type":
            "WELL_MATCH",
            "filter": {
                'species': content_tag['SPECIES'],
                'sex': content_tag['SEX'],
                'age': content_tag['AGE'],
            }
        })

        # PART_MATCH： （类别回退） 模糊搜索
        # for k, v in classes.items():
        #     r.append({
        #         "query": query.replace(k, v),
        #         "type": "PART_MATCH",
        #         "key_words": tags
        #     })

        return r

    def search(self, query):
        query = self.__preprocess(query, _clean=True, cut=False)
        if query == "":
            return {}

        seek_query_list = self.__query_analyze(query)

        result = self.bs.search(seek_query_list)
        # 计算摘要

        self.__abstract(seek_query_list, result)

        result = self.__format_to(result, "LIST")
        return result
