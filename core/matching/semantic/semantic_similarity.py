from core.matching import Matching
from config import get_cfg
from scipy import spatial
import numpy as np
from core.queryUnderstanding.representation import W2V, Embedding


class SemanticSimilarity(Matching):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.w2v = W2V(cfg)

    def M_cosine(self, s1_list, s2_list):
        v1 = Embedding.wam(s1_list, self.w2v.model, agg='mean')
        v2 = Embedding.wam(s2_list, self.w2v.model, agg='mean')
        sim = 1 - spatial.distance.cosine(v1, v2)
        return sim

    def get_score(self, s1, s2, model='cosine'):
        if model == 'cosine':
            f_ssim = self.M_cosine
        sim = f_ssim(s1, s2)
        return sim


if __name__ == '__main__':
    cfg = get_cfg()
    ss = SemanticSimilarity(cfg)
    s1 = ['哈士奇', '拆家', '怎么办']
    s2 = ['狗狗', '吐', '了', '怎么办']
    print(ss.get_score(s1, s2))