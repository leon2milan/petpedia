from qa.matching.lexical import LexicalSimilarity
from qa.matching.semantic import SemanticSimilarity
from qa.matching.semantic.simCSE import SIMCSE
from qa.tools import setup_logger
from qa.matching import Matching
from config import get_cfg

logger = setup_logger()


class Similarity(Matching):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.ls = LexicalSimilarity(cfg)
        self.ss = SemanticSimilarity(cfg)
        if 'simcse' in self.cfg.MATCH.METHODS:
            self.simcse = SIMCSE(cfg)

    def get_score_many(self, query, candidate_list):
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
        query_list = self.seg.cut(query)
        candidate_cut_list = self.seg.cut(candidate_list)
        res = []
        for i in range(len(candidate_cut_list)):
            score = self.get_score(query, candidate_list[i], query_list,
                                   candidate_cut_list[i])
            if len(scores) > 0:
                score += scores[i]

            res.append(score)
        return res

    def get_score(self, s1, s2, s1_list=None, s2_list=None):
        if s1_list is None or not isinstance(s1_list, list):
            s1_list = self.seg.cut(s1)
        if s2_list is None or not isinstance(s2_list, list):
            s2_list = self.seg.cut(s2)
        score = 0.0
        for method in self.cfg.MATCH.METHODS:
            if method == 'jaccard':
                score += self.ls.get_score(s1, s2, 'jaccard')
            elif method == 'edit':
                score += self.ls.get_score(s1, s2, 'edit')
            elif method == 'cosine':
                score += self.ss.get_score(s1_list, s2_list, 'cosine')
            elif method == 'ts_ss':
                score -= self.ss.get_score(s1_list, s2_list, 'ts_ss')
            elif method == 'ts':
                score += self.ss.get_score(s1_list, s2_list, 'ts')
            elif method == 'ss':
                score += self.ss.get_score(s1_list, s2_list, 'ss')
            elif method == 'simcse':
                score += 0.0
            else:
                logger.warning(
                    'We do not know the similarity method of {}. Please contact the developper.'
                    .format(method))
                score += 0
        return score


if __name__ == '__main__':
    cfg = get_cfg()
    ss = SemanticSimilarity(cfg)
    s1 = '哈士奇拆家怎么办'
    s2 = '狗狗吐了怎么办'
    print(ss.get_score(s1, s2))