import numpy as np
from config import get_cfg
from qa.matching import Matching
from qa.matching.lexical import LexicalSimilarity
from qa.matching.semantic import SemanticSimilarity
from qa.matching.semantic.simCSE import SIMCSE
from qa.tools import setup_logger

logger = setup_logger()
__all__ = ['Similarity']


class Similarity(Matching):
    __slot__ = ['cfg', 'ls', 'ss', 'simcse']

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.ls = LexicalSimilarity(cfg)
        self.ss = SemanticSimilarity(cfg)
        if 'simcse' in self.cfg.MATCH.METHODS:
            self.simcse = SIMCSE(cfg)

    def get_score_many(self, query, candidate_list, candidate_cut_list=None):
        if not candidate_list or not query:
            return []
        scores = []
        if 'simcse' in self.cfg.MATCH.METHODS:
            if isinstance(candidate_list, list):
                scores = self.simcse.query_sim(query, candidate_list)
            elif isinstance(candidate_list, str):
                scores = self.simcse.similarity(query, candidate_list)
            else:
                raise TypeError(
                    f"simCSE does not support {type(candidate_list)}")
        query_list = self.seg.cut(query) if isinstance(query, str) else query
        query = "".join(query) if isinstance(query, list) else query
        candidate_cut_list = self.seg.cut(
            candidate_list
        ) if candidate_cut_list is None else candidate_cut_list
        candidate_list = ["".join(x) for x in candidate_list]
        return self.get_score(query, candidate_list, query_list,
                              candidate_cut_list)

    def get_score(self, s1, s2, s1_list=None, s2_list=None):
        if s1_list is None or not isinstance(s1_list, list):
            s1_list = self.seg.cut(s1)
        if s2_list is None or not isinstance(s2_list, list):
            s2_list = self.seg.cut(s2)
        score = np.zeros((len(s2_list), ))
        for method in self.cfg.MATCH.METHODS:
            if method == 'jaccard':
                score += np.array(self.ls.get_score(s1, s2, 'jaccard'))
            elif method == 'edit':
                score += np.array(self.ls.get_score(s1, s2, 'edit'))
            elif method == 'cosine':
                score += self.ss.get_score(s1_list, s2_list, 'cosine')
            elif method == 'ts_ss':
                score -= self.ss.get_score(s1_list, s2_list, 'ts_ss')
            elif method == 'euclidean':
                score -= self.ss.get_score(s1_list, s2_list, 'euclidean')
            elif method == 'simcse':
                pass
            else:
                logger.warning(
                    'We do not know the similarity method of {}. Please contact the developper.'
                    .format(method))
        return score


if __name__ == '__main__':
    cfg = get_cfg()
    ss = SemanticSimilarity(cfg)
    s1 = '????????????????????????'
    s2 = '?????????????????????'
    print(ss.get_score(s1, s2))
