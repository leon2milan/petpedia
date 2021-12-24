from qa.matching import Matching
from config import get_cfg
from scipy import spatial
from scipy.special import softmax

import math
from qa.queryUnderstanding.representation import W2V, Embedding
import numpy as np
from qa.tools.utils import Singleton

__all__ = ['simical', 'SemanticSimilarity']


class simical():
    __slot__ = ['vec1', 'vec2']

    def __init__(self, vec1, vec2):
        self.vec1 = vec1
        self.vec2 = vec2

    def _Norm(self, vec):
        return np.linalg.norm(vec)

    def _Theta(self):
        return math.acos(self.Cosine()) + 10

    def _Magnitude_Difference(self):
        return abs(self._Norm(self.vec1) - self._Norm(self.vec2))

    def Euclidean(self):
        return spatial.distance.euclidean(self.vec1, self.vec2)

    def Cosine(self):
        return 1 - spatial.distance.cosine(self.vec1, self.vec2)

    def Triangle(self):
        # not a true conclusion
        theta = math.radians(self._Theta())
        return (self._Norm(self.vec1) * self._Norm(self.vec2) *
                math.sin(theta)) / 2

    def Sector(self):
        ED = self.Euclidean()
        MD = self._Magnitude_Difference()
        theta = self._Theta()
        return math.pi * math.pow((ED + MD), 2) * theta / 360

    def TS_SS(self):
        return self.Triangle() * self.Sector()


@Singleton
class SemanticSimilarity(Matching):
    __slot__ = ['cfg', 'rough_w2v', 'fine_w2v']

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rough_w2v = W2V(cfg, is_rough=True)
        self.fine_w2v = W2V(cfg)

    def delete_diff(self, query, is_rough=True):
        if len(query) == 1:
            return [1.0]
        model = self.rough_w2v.model if is_rough else self.fine_w2v.model
        base = Embedding.wam(query, model, agg='mean')
        res = np.array([
            1 -
            spatial.distance.cosine(Embedding.get_embedding(model, x), base)
            for x in query
        ])
        res = np.nan_to_num(res)
        return softmax(res)

    def get_score(self, s1, s2, mode='cosine', is_rough=True):
        model = self.rough_w2v.model if is_rough else self.fine_w2v.model
        v1 = Embedding.wam(s1, model, agg='mean')
        v2 = Embedding.wam(s2, model, agg='mean')
        sim = simical(v1, v2)
        if mode == 'cosine':
            f_ssim = sim.Cosine()
        elif mode == 'ts_ss':
            f_ssim = sim.TS_SS()
        elif mode == 'ts':
            f_ssim = sim.Triangle()
        elif mode == 'ss':
            f_ssim = sim.Sector()
        if math.isinf(f_ssim) or math.isnan(f_ssim):
            f_ssim = 0.0
        return f_ssim


if __name__ == '__main__':
    cfg = get_cfg()
    ss = SemanticSimilarity(cfg)
    s1 = ['哈士奇', '拆家', '怎么办']
    s2 = ['狗狗', '吐', '了', '怎么办']
    print(ss.get_score(s1, s2))
    print(ss.get_score(['聪明'], ['智商']))