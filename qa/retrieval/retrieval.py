import time

from config import get_cfg
from qa.matching import Similarity
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.retrieval.semantic.hnsw import HNSW
from qa.retrieval.term import TermRetrieval
from qa.tools import setup_logger
from qa.tools.utils import flatten

logger = setup_logger()


class BasicSearch():
    def __init__(self, cfg):
        logger.info('Initializing Basic Search Object ....')
        self.cfg = cfg
        self.seg = Segmentation(cfg)
        self.stopwords = Words(cfg).get_stopwords

        self.sim = Similarity(cfg)

        self.tr = TermRetrieval(cfg)
        self.rough_hnsw = HNSW(cfg, is_rough=True)
        self.fine_hnsw = HNSW(cfg, is_rough=False)

    def __add_score(self, score, weight):
        return score + weight

    def __avg_score(self, oldavg, w):
        """
        oldavg 之前的加权平均值
        w 新的因子
        """
        if w > 100000:
            w = 100000
        if oldavg < 1.0:
            return 2 + (oldavg + w) / 2.0
        count = int(oldavg)
        oldavgavg = oldavg - count
        return count + 1 + (oldavgavg * count + w) / (count + 1)

    def __remove_invalid_doc(self, hitlist):
        if self.cfg.RETRIEVAL.USE_ES:
            return hitlist
        r = []
        if hitlist:
            for i in hitlist:
                if i['docid'] not in self.tr.ri._invalid_docid:
                    r.append(i)
        return r

    def __remove_mustnot(self, hitlist, notlist):
        if len(notlist) == 0:
            return hitlist
        notmap = {}
        for i in notlist:
            notmap[i["docid"]] = 1

        r = []
        for i in hitlist:
            if i["docid"] in notmap:
                continue
            r.append(i)
        return r

    def __trans_to_ind(self, sort_doc_list):
        if sort_doc_list is None:
            return []
        if self.cfg.RETRIEVAL.USE_ES:
            return sort_doc_list
        for i in sort_doc_list:
            if i["docid"] in self.tr.ri._docid_name_map:
                i["doc"] = self.tr.ri._docid_name_map[i["docid"]]["doc"]
        return sort_doc_list

    def __filter_duplicate(self, result):
        res = []
        exist = {}
        matchtype = "BEST_MATCH"
        if matchtype in result:
            for i in result[matchtype]:
                if i["docid"] not in exist:
                    i['match_type'] = matchtype
                    res.append(i)
                    exist[i["docid"]] = 1

        matchtype = "WELL_MATCH"
        if matchtype in result:
            for i in result[matchtype]:
                if i["docid"] not in exist:
                    i['match_type'] = matchtype
                    res.append(i)
                    exist[i["docid"]] = 1

        matchtype = "PART_MATCH"
        if matchtype in result:
            for i in result[matchtype]:
                if i["docid"] not in exist:
                    i['match_type'] = matchtype
                    res.append(i)
                    exist[i["docid"]] = 1
        return res

    def __preprocess(self, query, is_rough=False):
        query = query.strip()
        if query == "":
            return ""
        query = [
            (x[0], x[-1])
            for x in zip(*self.seg.cut(query, mode='rank', is_rough=is_rough))
            # if x[-1] > 1 and x[0] not in self.stopwords
            if x[0] not in self.stopwords
        ]
        return query

    def __keywords_filter(self, result, keywords):
        # result = [
        #     x for x in result
        #     if x and any(keyword in x['docid'] for keyword in keywords)
        # ]
        if 'DOG' in keywords and 'CAT' not in keywords:
            result = [x for x in result if '猫' not in x['docid']]
        elif 'DOG' not in keywords and 'CAT' in keywords:
            result = [x for x in result if '狗' not in x['docid']]
        return result

    def search(self, seek_query_list):
        if seek_query_list == []:
            return {}

        result = {}
        result["BEST_MATCH"] = []
        result["WELL_MATCH"] = []
        result["PART_MATCH"] = []
        notresult = []
        donequery = {}
        for query in seek_query_list:
            if (query["type"] == "BEST_MATCH"
                    and query["raw"]) or (query["type"] != "BEST_MATCH"):
                rough_query = self.__preprocess(query["query"], is_rough=True)
                fine_query = self.__preprocess(query["query"], is_rough=False)
            querysign = "%s%s" % (query["query"], query["type"])
            if querysign in donequery:
                continue
            donequery[querysign] = 1
            logger.debug(
                "query 检索原语: {}, rough cut 后 {}, fine cut 后 {}".format(
                    query, rough_query if rough_query is not None else '',
                    fine_query if fine_query is not None else ''))

            if query["type"] == "BEST_MATCH":
                result["BEST_MATCH"] += self.tr.search(query["query"],
                                                       query['type'],
                                                       is_rough=True)
                if query["raw"]:
                    result["BEST_MATCH"] += flatten(
                        self.rough_hnsw.search([x[0] for x in rough_query]))

                    result["BEST_MATCH"] += flatten(
                        self.fine_hnsw.search([x[0] for x in fine_query]))

            if query["type"] == "WELL_MATCH":
                result["WELL_MATCH"] += self.tr.search(rough_query,
                                                       query['type'],
                                                       is_rough=True)
                if len(result["WELL_MATCH"]) < self.cfg.RETRIEVAL.LIMIT:
                    result["WELL_MATCH"] += self.tr.search(fine_query,
                                                           query['type'],
                                                           is_rough=False)

            if query["type"] == "PART_MATCH":
                if len(result["BEST_MATCH"]) + len(result["WELL_MATCH"]) < int(
                        self.cfg.RETRIEVAL.PART_MATCH_RATIO *
                        self.cfg.RETRIEVAL.LIMIT):
                    result["PART_MATCH"] += self.tr.search(rough_query,
                                                           query['type'],
                                                           is_rough=True)

                    if len(result["PART_MATCH"]) < self.cfg.RETRIEVAL.LIMIT:
                        result["PART_MATCH"] += self.tr.search(fine_query,
                                                               query['type'],
                                                               is_rough=False)
            logger.debug('retrieval result befort filter: {}'.format(result))

            # 根据关键词过滤
            result[query["type"]] = self.__keywords_filter(
                result[query["type"]], query["key_words"])
            logger.debug('retrieval result after filter: {}'.format(result))

            # 减法
            if "mustnot" in query:
                notresult += self.tr.search(self.__preprocess(query["mustnot"],
                                                              is_rough=True),
                                            mode='well',
                                            is_rough=True,
                                            limit=1000000)
                notresult += self.tr.search(self.__preprocess(query["mustnot"],
                                                              is_rough=False),
                                            mode='well',
                                            is_rough=False,
                                            limit=1000000)
                notresult += self.tr.search(self.__preprocess(query["mustnot"],
                                                              is_rough=True),
                                            mode='part',
                                            is_rough=True,
                                            limit=1000000)
                notresult += self.tr.search(self.__preprocess(query["mustnot"],
                                                              is_rough=False),
                                            mode='part',
                                            is_rough=False,
                                            limit=1000000)

        # l0 截断排序
        for match_type in result:
            result[match_type] = self.__remove_mustnot(result[match_type],
                                                       notresult)
            result[match_type] = self.__remove_invalid_doc(result[match_type])
            result[match_type] = TermRetrieval.sort_result(result[match_type])
            result[match_type] = TermRetrieval.limit(result[match_type],
                                                     self.cfg.RETRIEVAL.LIMIT)
        # 去重
        result = self.__filter_duplicate(result)
        logger.debug('retrieval result after deduplicated: {}'.format(result))

        # l1 相关性排序 粗排
        s = time.time()
        result = self.__cal_similarity([
            x['query'] for x in seek_query_list if x['type'] == 'BEST_MATCH'
        ][0], result)
        logger.debug('Cal simlarity takes: {}'.format(time.time() - s))

        result = TermRetrieval.sort_result(result)
        result = TermRetrieval.limit(result, self.cfg.RETRIEVAL.LIMIT)
        result = self.__trans_to_ind(result)
        logger.debug('retrieval final result: {}'.format(result))

        return result

    def __cal_similarity(self, query, result):
        s = time.time()
        scores = self.sim.get_score_many(
            query, [x['docid'].split(':')[0] for x in result])
        logger.debug('Get score takes: {}'.format(time.time() - s))
        for i in range(len(result)):
            result[i]['score'] = scores[i]
        return result


if __name__ == "__main__":
    cfg = get_cfg()
    bs = BasicSearch(cfg)
    s = [{
        'query': '哈士奇拆家怎么办？',
        'type': 'BEST_MATCH'
    }, {
        'query': '哈士奇拆家怎么办？',
        'type': 'WELL_MATCH'
    }, {
        'query': '哈士奇拆家怎么办？',
        'type': 'PART_MATCH'
    }]
    bs.search(s)
