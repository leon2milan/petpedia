import time

from config import get_cfg
from qa.matching import Similarity
from qa.retrieval.semantic.hnsw import HNSW
from qa.retrieval.term import TermRetrieval
from qa.tools import setup_logger
from qa.tools.utils import flatten
from qa.retrieval.manual.manual import Manual

from joblib import Parallel, delayed
import dill 

logger = setup_logger()
__all__ = ['BasicSearch']


def unwrap_self(arg, **kwarg):
    return BasicSearch.search_helper(*arg, **kwarg)


class BasicSearch():
    __slot__ = [
        'cfg', 'sim', 'tr', 'rough_hnsw', 'fine_hnsw', 'rules'
    ]

    def __init__(self, cfg):
        logger.debug('Initializing Basic Search Object ....')
        self.cfg = cfg

        self.sim = Similarity(cfg)

        self.tr = TermRetrieval(cfg)
        self.rough_hnsw = HNSW(cfg, is_rough=True)
        self.fine_hnsw = HNSW(cfg, is_rough=False)
        self.rules = Manual(self.cfg).get_rule()

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

    def __keywords_filter(self, result, _filter):
        for name, item in self.rules.items():
            if _filter[name]:
                for i in item:
                    rule = i['rule']
                    result = [
                        x for x in result if x['tag'][name] and x['tag'][name]
                        in [r for r in rule if r in _filter[name]]
                    ]
        return result

    def _mod_score_by_term_weight(self, query_terms, hitmap, is_rough=True):
        if not hitmap:
            return hitmap
        columns = 'question_rough_cut' if is_rough else 'question_fine_cut'
        query_terms = {k: v for k, v in query_terms}
        for i in range(len(hitmap)):
            hitmap[i]['score'] = sum([
                query_terms.get(word, 0.0) * 20
                for word in hitmap[i]['doc'][columns]
            ])
        return hitmap

    def _mod_score_by_close_weight(self,
                                   query_terms,
                                   result_list,
                                   is_rough=True):
        columns = 'question_rough_cut' if is_rough else 'question_fine_cut'
        query_terms = [x[0] for x in query_terms]
        # 计算term之间紧密度
        for i in result_list:
            pos_list = [
                i for i, w in enumerate(i['doc'][columns]) if w in query_terms
            ]
            close_weight = self.__close_weight([pos_list])
            i["score"] = self.__add_score(i["score"], close_weight)
        return result_list

    def __bestmatch_change_score(self, query, hitlist):
        for i in hitlist:
            i["score"] = self.__add_score(i["score"], 2.0)
        return hitlist

    def __add_score(self, score, weight):
        return score + weight

    def sort_result(self, hitlist):
        if hitlist is None:
            return []
        # sort hitlist by score
        return sorted(hitlist, key=lambda key: key['score'], reverse=True)

    def limit(self, hitlist, limit):
        if len(hitlist) > limit:
            return hitlist[:limit]
        return hitlist

    def __close_weight(self, poslist):
        """
        poslist is a list
        [0] -> set(pos1, pos2...)
        [1] -> set()
        return:
            a float [0, 5]
            数值越大, 意味着应该更大提权
        """
        if poslist is None:
            return 0
        if len(poslist) == 0:
            return 0

        step = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 100, 200, 400, 1000]
        """
        把每个 query 的命中位置对齐到不同粒度的位置
        所有 set 之间交集不为空视为最小紧密度
        """
        all_close_weight = None  # 所有 term 之间最大距离
        for curstep in step:
            normalize_sets = []
            for s in poslist:
                normalize_sets.append(set())
                for p in s:
                    mod = p % curstep
                    lowbound = p - mod
                    highbound = p + (curstep - mod)
                    normalize_sets[-1].add(lowbound)
                    normalize_sets[-1].add(highbound)

            ## 求所有集合交集
            and_set = None
            for s in normalize_sets:
                if and_set is None:
                    and_set = s
                    continue
                and_set = and_set & s
            if and_set and len(and_set) > 0:
                all_close_weight = curstep

            if all_close_weight:
                return (2.0) / all_close_weight
        """
        上面是全部term都比较临近
        下面是部分term临近
        """
        distance = []
        for curstep in step:
            part_close_weight = 0
            normalize_sets = []
            for s in poslist:
                normalize_sets.append(set())
                for p in s:
                    mod = p % curstep
                    lowbound = p - mod
                    highbound = p + (curstep - mod)
                    normalize_sets[-1].add(lowbound)
                    normalize_sets[-1].add(highbound)

            ## 求任意集合的交集
            or_set = None
            for s in normalize_sets:
                if or_set is None:
                    or_set = s
                    continue
                if len(or_set & s) > 0:
                    part_close_weight += curstep
                    or_set = or_set | s
                else:
                    part_close_weight += 100
                    or_set = or_set | s
            if part_close_weight > 0:
                distance.append(part_close_weight)
        distance.sort()
        if len(distance):
            return 1.5 / (distance[0])
        return 0

    def __post_best_match(self, query, hitlist, limit=None):
        # no word seg
        if limit is None:
            limit = self.cfg.RETRIEVAL.LIMIT

        hitlist = self.__bestmatch_change_score(query, hitlist)
        sort_docid = self.sort_result(hitlist)
        sort_docid = self.limit(sort_docid, limit)
        logger.debug("best 检索结果数: %s" % len(sort_docid))
        return sort_docid

    def __post_well_match(self, query, hitlist, limit=None):
        # with word seg
        # all term must hit
        if limit is None:
            limit = self.cfg.RETRIEVAL.LIMIT

        # 根据term 重要性调整分数
        if self.cfg.RETRIEVAL.DO_TERM_WEIGHT:
            hitlist = self._mod_score_by_term_weight(query, hitlist)

        # term紧密度
        if self.cfg.RETRIEVAL.DO_CLOSE_WEIGHT:
            hitlist = self._mod_score_by_close_weight(query, hitlist)
        # 排序
        sort_docid = self.sort_result(hitlist)
        # 截断
        sort_docid = self.limit(sort_docid, limit)
        logger.debug("well 检索结果数: %s" % len(sort_docid))
        return sort_docid

    def __post_partmatch(self, query, hitlist, limit=None):
        # with word seg
        # some term can missmatch
        if limit is None:
            limit = self.cfg.RETRIEVAL.LIMIT

        # 根据term 重要性调整分数
        if self.cfg.RETRIEVAL.DO_TERM_WEIGHT:
            hitlist = self._mod_score_by_term_weight(query, hitlist)

        # term紧密度
        if self.cfg.RETRIEVAL.DO_CLOSE_WEIGHT:
            hitlist = self._mod_score_by_close_weight(query, hitlist)
        # 排序
        sort_docid = self.sort_result(hitlist)
        # 截断
        sort_docid = self.limit(sort_docid, limit)
        logger.debug("part 检索结果数: %s" % len(sort_docid))
        return sort_docid

    def search_helper(self, query):
        if not query.get('rough_query') and not query.get('fine_query'):
            result = []
        
        logger.debug(
            "query 检索原语: {}, rough cut 后 {}, fine cut 后 {}".format(
                query['query'], query.get('rough_query'),
                query.get('fine_query')))
        s = time.time()
        if query["type"] == "BEST_MATCH":
            tmp = []
            if self.cfg.RETRIEVAL.BEST_ROUTE.DO_KBQA:
                pass

            if self.cfg.RETRIEVAL.WELL_ROUTE.USE_ES:
                tmp += self.tr.search(query['query'], query['rough_query'],
                                      query['fine_query'], query['type'])

            result = self.__post_best_match(query["query"],
                                            tmp,
                                            limit=self.cfg.RETRIEVAL.LIMIT)

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('BEST_MATCH takes: {}'.format(time.time() - s))

        s = time.time()
        if query["type"] == "WELL_MATCH":
            tmp = []
            term_weight = self.sim.ss.delete_diff(query['rough_query'])
            if query['rough_query'] is not None and query['rough_query']:
                tmp += flatten(self.rough_hnsw.search(query['rough_query']))

            if query['fine_query'] is not None and query['fine_query']:
                tmp += flatten(self.fine_hnsw.search(query['fine_query']))

            if self.cfg.RETRIEVAL.WELL_ROUTE.USE_ES:
                tmp += self.tr.search(query['query'], query['rough_query'],
                                      query['fine_query'], query['type'])
            result = self.__post_well_match(list(
                zip(query['rough_query'], term_weight)),
                                            tmp,
                                            limit=self.cfg.RETRIEVAL.LIMIT)

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('WELL_MATCH takes: {}'.format(time.time() - s))

        s = time.time()
        if query["type"] == "PART_MATCH":
            if self.cfg.RETRIEVAL.PART_ROUTE.USE_CONTENT_PROFILE:
                pass

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('PART_MATCH takes: {}'.format(time.time() - s))
        return {'type': query['type'], 
                'result': self.__keywords_filter(result, query["filter"])}

    def search2(self, seek_query_list):
        if seek_query_list == []:
            return {}

        result = {}
        result["BEST_MATCH"] = []
        result["WELL_MATCH"] = []
        result["PART_MATCH"] = []
        notresult = []

        results = Parallel(n_jobs=2, backend='threading')(
            delayed(unwrap_self)((self, query)) for query in seek_query_list)
        for res in results:
            if res['type'] == 'BEST_MATCH':
                result["BEST_MATCH"] += res['result']
            elif res['type'] == 'WELL_MATCH':
                result["WELL_MATCH"] += res['result']
            elif res['type'] == 'PART_MATCH':
                result["PART_MATCH"] += res['result']

        s = time.time()
        # l0 截断
        for match_type in result:
            result[match_type] = self.__remove_mustnot(result[match_type],
                                                       notresult)
            result[match_type] = self.__remove_invalid_doc(result[match_type])

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('__remove takes: {}'.format(time.time() - s))

        # 去重
        s = time.time()
        result = self.__filter_duplicate(result)
        logger.debug('retrieval result after deduplicated: {}'.format(result))

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('__filter_duplicate takes: {}'.format(time.time() - s))

        # l1 相关性排序 粗排
        s = time.time()
        result = self.__cal_similarity([
            x['rough_query'] for x in seek_query_list
            if x['type'] == 'BEST_MATCH'
        ][0], result)

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('Cal simlarity takes: {}'.format(time.time() - s))

        s = time.time()
        result = self.sort_result(result)
        result = self.limit(result, self.cfg.RETRIEVAL.LIMIT)
        logger.debug('retrieval final result: {}'.format(result))

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('sort_result takes: {}'.format(time.time() - s))

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
            querysign = "%s%s" % (query["query"], query["type"])
            if querysign in donequery or (not query.get('rough_query')
                                          and not query.get('fine_query')):
                continue
            donequery[querysign] = 1
            logger.debug(
                "query 检索原语: {}, rough cut 后 {}, fine cut 后 {}".format(
                    query['query'], query.get('rough_query'),
                    query.get('fine_query')))

            s = time.time()
            if query["type"] == "BEST_MATCH":
                tmp = []
                if self.cfg.RETRIEVAL.BEST_ROUTE.DO_KBQA:
                    pass

                if self.cfg.RETRIEVAL.WELL_ROUTE.USE_ES:
                    tmp += self.tr.search(query['query'], query['rough_query'],
                                          query['fine_query'], query['type'])

                result["BEST_MATCH"] += self.__post_best_match(
                    query["query"], tmp, limit=self.cfg.RETRIEVAL.LIMIT)

            if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                logger.info('BEST_MATCH takes: {}'.format(time.time() - s))

            s = time.time()
            if query["type"] == "WELL_MATCH":
                term_weight = self.sim.ss.delete_diff(query['rough_query'])
                if query['rough_query'] is not None and query['rough_query']:

                    tmp = flatten(self.rough_hnsw.search(query['rough_query']))
                    result["WELL_MATCH"] += self.__post_well_match(
                        list(zip(query['rough_query'], term_weight)),
                        tmp,
                        limit=self.cfg.RETRIEVAL.LIMIT)
                    if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                        logger.info('rough hnsw WELL_MATCH takes: {}'.format(
                            time.time() - s))

                # if query['fine_query'] is not None and query['fine_query']:
                #     tmp = flatten(self.fine_hnsw.search(query['fine_query']))

                #     result["WELL_MATCH"] += self.__post_well_match(
                #         list(zip(query['rough_query'], term_weight)),
                #         tmp,
                #         limit=self.cfg.RETRIEVAL.LIMIT)
                #     if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                #         logger.info('fine hnsw WELL_MATCH takes: {}'.format(
                #             time.time() - s))

                if self.cfg.RETRIEVAL.WELL_ROUTE.USE_ES:
                    tmp = self.tr.search(query['query'], query['rough_query'],
                                         query['fine_query'], query['type'])
                    result["WELL_MATCH"] += self.__post_well_match(
                        list(zip(query['rough_query'], term_weight)),
                        tmp,
                        limit=self.cfg.RETRIEVAL.LIMIT)
                    if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                        logger.info('es takes: {}'.format(time.time() - s))

            if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                logger.info('WELL_MATCH takes: {}'.format(time.time() - s))

            s = time.time()
            if query["type"] == "PART_MATCH":
                if self.cfg.RETRIEVAL.PART_ROUTE.USE_CONTENT_PROFILE:
                    pass

            if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                logger.info('PART_MATCH takes: {}'.format(time.time() - s))
            logger.debug('retrieval result befort filter: {}'.format(result))

            # 根据关键词过滤
            s = time.time()
            result[query["type"]] = self.__keywords_filter(
                result[query["type"]], query["filter"])
            logger.debug('retrieval result after filter: {}'.format([result]))

            if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
                logger.info('__keywords_filter takes: {}'.format(time.time() -
                                                                 s))

            logger.debug('{}, takes: {}'.format(
                query['type'], [x['docid'] for x in result[query['type']]]))

            # 减法
            if "mustnot" in query:
                pass

        s = time.time()
        # l0 截断
        for match_type in result:
            result[match_type] = self.__remove_mustnot(result[match_type],
                                                       notresult)
            result[match_type] = self.__remove_invalid_doc(result[match_type])

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('__remove takes: {}'.format(time.time() - s))

        # 去重
        s = time.time()
        result = self.__filter_duplicate(result)
        logger.debug('retrieval result after deduplicated: {}'.format(result))

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('__filter_duplicate takes: {}'.format(time.time() - s))

        # l1 相关性排序 粗排
        s = time.time()
        result = self.__cal_similarity([
            x['rough_query'] for x in seek_query_list
            if x['type'] == 'BEST_MATCH'
        ][0], result)

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('Cal simlarity takes: {}'.format(time.time() - s))

        s = time.time()
        result = self.sort_result(result)
        result = self.limit(result, self.cfg.RETRIEVAL.LIMIT)
        logger.debug('retrieval final result: {}'.format(result))

        if self.cfg.RETRIEVAL.TIME_PERFORMENCE:
            logger.info('sort_result takes: {}'.format(time.time() - s))

        return result

    def __cal_similarity(self, query, result):
        s = time.time()
        scores = self.sim.get_score_many(
            query, [x['doc']['question'] for x in result], [
                x.split() for x in
                [" ".join(x['doc']['question_rough_cut']) for x in result]
            ])
        logger.debug('query: {}, befor rank result is : {}'.format(
            query, [" ".join(x['doc']['question_rough_cut']) for x in result]))
        assert len(result) == len(
            scores
        ), f"result length: {len(result)} doesn't match scores length {len(scores)}"
        logger.debug('Get score takes: {}'.format(time.time() - s))
        for i in range(len(result)):
            score = scores[i]

            result[i]['score'] = score
        return result


if __name__ == "__main__":
    cfg = get_cfg()
    bs = BasicSearch(cfg)
    query = [{
        'query': '犬细小',
        'rough_query': ['犬细小'],
        'fine_query': ['犬', '细小'],
        'query': '犬细小',
        'type': 'BEST_MATCH',
        'filter': {
            'species': ['DOG'],
            'sex': [],
            'age': []
        }
    }, {
        'query': '犬细小',
        'rough_query': ['犬细小'],
        'fine_query': ['犬', '细小'],
        'type': 'WELL_MATCH',
        'filter': {
            'species': ['DOG'],
            'sex': [],
            'age': []
        }
    }, {
        'query': '犬细小',
        'rough_query': ['犬细小'],
        'fine_query': ['犬', '细小'],
        'type': 'PART_MATCH',
        'filter': {
            'species': ['DOG'],
            'sex': [],
            'age': []
        }
    }]
    import time
    s = time.time()
    print('res: ', [(x['score'], x['docid']) for x in bs.search(query)],
          'time: ',
          time.time() - s)
