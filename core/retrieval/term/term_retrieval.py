import math
import hashlib

from config import get_cfg
from core.tools import setup_logger

logger = setup_logger()


class TermRetrieval():
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.RETRIEVAL.USE_ES:
            from core.tools.es import ES
            self.es = ES(cfg)
        else:
            from core.server import Index
            self.ri = Index(self.cfg)
            self.K1 = self.cfg.RETRIEVAL.TERM.K1
            self.K3 = self.cfg.RETRIEVAL.TERM.K3
            self.B = self.cfg.RETRIEVAL.TERM.B

    def search(self, query, mode, is_rough, limit=None):
        if mode == 'BEST_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, is_rough, is_exact=True)
            else:
                return self.__search_best_match(query,
                                                is_rough=is_rough,
                                                limit=limit)
        elif mode == 'WELL_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, is_rough)
            else:
                return self.__search_well_match(query,
                                                is_rough=is_rough,
                                                limit=limit)
        elif mode == 'PART_MATCH':
            if self.cfg.RETRIEVAL.USE_ES:
                return self.__seek_es(query, is_rough)
            else:
                return self.__search_partmatch(query,
                                               is_rough=is_rough,
                                               limit=limit)
        else:
            raise NotImplementedError

    def __seek_es(self, term, is_rough, is_exact=False):
        row = 'question_rough_cut' if is_rough else 'question_fine_cut'
        term = " ".join([x[0] for x in term]) if isinstance(term, list) else term
        if is_exact:
            return self.es.exact_search(self.cfg.RETRIEVAL.ES_INDEX,
                                        'question', term)
        else:
            return self.es.fuzzy_search(self.cfg.RETRIEVAL.ES_INDEX, row, term)

    def __seek_one_term(self, term, is_rough):
        # return {} -> [dockid]: {"docid":, "score":}
        termhash = hashlib.md5(term.encode(encoding='utf-8')).hexdigest()
        r = {}
        AVG_L = self.ri.get_rough_l() if is_rough else self.ri.get_fine_l()
        if self.ri.get_rindex(termhash, is_rough) is not None:
            for docid in self.ri.get_rindex(termhash, is_rough):
                TF = self.ri.get_tf(termhash, docid, is_rough)
                DF = len(self.ri.get_rindex(termhash, is_rough))
                LD = self.ri.get_ld(termhash, docid, is_rough)
                IDF = math.log2(
                    (self.ri.get_doc_num() - DF + 0.5) / (DF + 0.5))
                # TODO
                # add term-query relational calculation (k3 + 1) * tf_tq / (k3 + tf_tq)
                score = (self.K1 * TF *
                         IDF) / (TF + self.K1 *
                                 (1 - self.B + self.B * LD / AVG_L))
                r[docid] = {
                    "score": score,
                    "pos": self.ri.get_pos(termhash, docid, is_rough),
                    "docid": docid
                }
        return r

    def _mod_score_by_term_weight(self, query_terms, hitmap):
        for term, imp in query_terms:
            if term in hitmap.keys():
                hitmap[term]['score'] = imp * hitmap[term]['score']
        return hitmap

    def __merge_term_or(self, hitmap):
        res = {}
        for term in hitmap:
            for docid in hitmap[term]:
                if docid not in res:
                    res[docid] = {}
                    res[docid]["score"] = 0
                    res[docid]["poslist"] = {}
                    res[docid]["docid"] = ""
                    res[docid]["hitterm"] = {}
                # score 累加
                res[docid]["score"] += hitmap[term][docid]["score"]
                # 所有term位置
                res[docid]["poslist"][term[0]] = hitmap[term][docid]["pos"]
                # 文档id
                res[docid]["docid"] = docid
                # 该文档命了哪些term
                res[docid]["hitterm"][term[0]] = 1
        return res.values()

    def __merge_term_and(self, hitmap):
        """
        求取每个关键词的文档 交集
        并且多个词命中的相同文档, 需要计算紧密度
        Returns:
            return a list, 
            res["score"] = 0.3
            res["poslist"] = {term: [pos1, pos2], term2:[pos3, pos4]} 
            res["docid"] = "" 
            res["hitterm"] = {term:1, term:1, term:1} 
        """

        ## 文档交集列表, 仅 docid
        and_set = None
        for term in hitmap:
            if and_set is None:
                and_set = set(hitmap[term])
                continue
            and_set = and_set & set(hitmap[term])
        # 初始化交集文档的初始score 值
        res = {}
        for term in hitmap:
            for docid in hitmap[term]:
                if docid not in and_set:
                    continue
                if docid not in res:
                    res[docid] = {}
                    res[docid]["score"] = 0
                    res[docid]["poslist"] = {}
                    res[docid]["docid"] = ""
                    res[docid]["hitterm"] = {}
                # score 累加
                res[docid]["score"] += hitmap[term][docid]["score"]
                # 所有term位置
                res[docid]["poslist"][term[0]] = hitmap[term][docid]["pos"]
                # 文档id
                res[docid]["docid"] = docid
                # 该文档命了哪些term
                res[docid]["hitterm"][term[0]] = 1

        return res.values()

    def _get_term_weight(self, query_terms):
        ori_terms = {}
        for i in query_terms:
            ori_terms[i] = 1
        return ori_terms

    def _mod_score_by_term_count(self, query_terms, result_list):
        ori_terms = self._get_term_weight(query_terms)
        fullcount = len(ori_terms) * 1.0
        for i in result_list:
            # 命中term的数量
            hit_count = len(i["hitterm"])
            add_score = 1.0 * hit_count
            i["score"] = self.__add_score(i["score"], add_score)
        return result_list

    def _mod_score_by_close_weight(self, result_list):
        # 计算term之间紧密度
        for i in result_list:
            pos_list = i["poslist"].values()
            close_weight = self.__close_weight(pos_list)
            i["score"] = self.__add_score(i["score"], close_weight)
        return result_list

    def __bestmatch_change_score(self, query, hitlist):
        ratio = len(query.split(' '))
        for i in hitlist:
            i["score"] = self.__add_score(i["score"], 2.0)
        return hitlist

    def __add_score(self, score, weight):
        return score + weight

    @staticmethod
    def sort_result(hitlist):
        if hitlist is None:
            return []
        # sort hitlist by score
        return sorted(hitlist, key=lambda key: key['score'], reverse=True)

    @staticmethod
    def limit(hitlist, limit):
        if len(hitlist) > limit:
            return hitlist[:limit]
        return hitlist

    def __search_best_match(self, query, is_rough, limit=None):
        # no word seg
        # query_result = {"score":, "pos":, "docid"}
        if limit is None:
            limit = self.cfg.RETRIEVAL.LIMIT
        query_result = self.__seek_one_term(query, is_rough=is_rough)
        if len(query_result) == 0:
            return []

        hitlist = query_result.values()
        hitlist = self.__bestmatch_change_score(query, hitlist)
        sort_docid = TermRetrieval.sort_result(hitlist)
        sort_docid = TermRetrieval.limit(sort_docid, limit)
        logger.debug("best 检索结果数: %s" % len(sort_docid))
        return sort_docid

    def __search_well_match(self, query, is_rough, limit=None):
        # with word seg
        # all term must hit
        if limit is None:
            limit = self.cfg.RETRIEVAL.LIMIT
        query_terms = query.split(" ") if isinstance(query, str) else query
        result = {}
        for i in query_terms:
            # query_terms: [(term, importance)...]
            # result[i] = {"score":, "pos":, "docid"}
            result[i] = self.__seek_one_term(i[0], is_rough=is_rough)

        # 根据term 重要性调整分数
        result = self._mod_score_by_term_weight(query_terms, result)
        # 拉链归并
        hitlist = self.__merge_term_and(result)
        # term匹配数调整权值, 匹配3个词 > 匹配2个词 > 匹配一个词
        hitlist = self._mod_score_by_term_count(query_terms, hitlist)
        # term紧密度
        hitlist = self._mod_score_by_close_weight(hitlist)
        # 排序
        sort_docid = TermRetrieval.sort_result(hitlist)
        # 截断
        sort_docid = TermRetrieval.limit(sort_docid, self.cfg.RETRIEVAL.LIMIT)
        logger.debug("well 检索结果数: %s" % len(sort_docid))
        return sort_docid

    def __search_partmatch(self, query, is_rough, limit=None):
        # with word seg
        # some term can missmatch
        if limit is None:
            limit = self.cfg.RETRIEVAL.LIMIT
        query_terms = query.split(" ") if isinstance(query, str) else query
        result = {}
        for i in query_terms:
            # query_terms: [(term, importance)...]
            # result[i] = {"score":, "pos":, "docid"}
            result[i] = self.__seek_one_term(i[0], is_rough=is_rough)
            if len(result[i]) == 0:
                del result[i]

        # 根据term 重要性调整分数
        result = self._mod_score_by_term_weight(query_terms, result)

        hitlist = self.__merge_term_or(result)

        # term匹配数调整权值, 匹配3个词 > 匹配2个词 > 匹配一个词
        hitlist = self._mod_score_by_term_count(query_terms, hitlist)
        # term紧密度
        hitlist = self._mod_score_by_close_weight(hitlist)
        # 排序
        sort_docid = TermRetrieval.sort_result(hitlist)
        # 截断
        sort_docid = TermRetrieval.limit(sort_docid, limit)
        logger.debug("part 检索结果数: %s" % len(sort_docid))
        return sort_docid

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


if __name__ == '__main__':
    cfg = get_cfg()
    tr = TermRetrieval(cfg)
