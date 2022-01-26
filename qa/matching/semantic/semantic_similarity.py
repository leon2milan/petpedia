import math

import numpy as np
from config import get_cfg
from qa.matching import Matching
from qa.queryUnderstanding.representation import W2V, Embedding
from qa.tools.utils import Singleton
import scipy

__all__ = ['euclidean', 'SemanticSimilarity']


class Simcal():
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
        return scipy.spatial.distance.euclidean(self.vec1, self.vec2)

    def Cosine(self):
        return 1 - scipy.spatial.distance.cosine(self.vec1, self.vec2)

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


# @nb.jit(nopython=True, fastmath=True)
# def init_w(w, n):
#     """
#     :purpose:
#     Initialize a weight array consistent of 1s if none is given
#     This is called at the start of each function containing a w param
#     :params:
#     w      : a weight vector, if one was given to the initial function, else None
#              NOTE: w MUST be an array of np.float64. so, even if you want a boolean w,
#              convert it to np.float64 (using w.astype(np.float64)) before passing it to
#              any function
#     n      : the desired length of the vector of 1s (often set to len(u))
#     :returns:
#     w      : an array of 1s with shape (n,) if w is None, else return w un-changed
#     """
#     if w is None:
#         return np.ones(n)
#     else:
#         return w

# @nb.jit(nopython=True, fastmath=True)
# def cosine(u, v, w=None):
#     """
#     :purpose:
#     Computes the cosine similarity between two 1D arrays
#     Unlike scipy's cosine distance, this returns similarity, which is 1 - distance
#     :params:
#     u, v   : input arrays, both of shape (n,)
#     w      : weights at each index of u and v. array of shape (n,)
#              if no w is set, it is initialized as an array of ones
#              such that it will have no impact on the output
#     :returns:
#     cosine  : float, the cosine similarity between u and v
#     :example:
#     >>> import numpy as np
#     >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
#     >>> cosine(u, v, w)
#     0.7495065944399267
#     """
#     n = len(u)
#     w = init_w(w, n)
#     num = 0
#     u_norm, v_norm = 0, 0
#     for i in range(n):
#         num += u[i] * v[i] * w[i]
#         u_norm += abs(u[i])**2 * w[i]
#         v_norm += abs(v[i])**2 * w[i]

#     denom = (u_norm * v_norm)**(1 / 2)
#     return num / denom

# @nb.jit(nopython=True, fastmath=True)
# def cosine_vector_to_matrix(u, m):
#     """
#     :purpose:
#     Computes the cosine similarity between a 1D array and rows of a matrix
#     :params:
#     u      : input vector of shape (n,)
#     m      : input matrix of shape (m, n)
#     :returns:
#     cosine vector  : np.array, of shape (m,) vector containing cosine similarity between u
#                      and the rows of m
#     :example:
#     >>> import numpy as np
#     >>> u = np.random.RandomState(seed=0).rand(10)
#     >>> m = np.random.RandomState(seed=0).rand(100, 10)
#     >>> cosine_vector_to_matrix(u, m)
#     (returns an array of shape (100,))
#     """
#     norm = 0
#     for i in range(len(u)):
#         norm += abs(u[i])**2
#     u = u / norm**(1 / 2)
#     for i in range(m.shape[0]):
#         norm = 0
#         for j in range(len(m[i])):
#             norm += abs(m[i][j])**2
#         m[i] = m[i] / norm**(1 / 2)
#     return np.dot(u, m.T)


def euclidean(x1, x2):
    return np.einsum('ij,ij->i', x1, x1)[:, np.newaxis] + np.einsum(
        'ij,ij->i', x2, x2) - 2 * np.dot(x1, x2.T)


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
        res = [Embedding.get_embedding(model, x).reshape(1, -1) for x in query]
        res = np.array([
            0.0 if np.all(
                x == 0) else scipy.spatial.distance.cdist(x, base, 'cosine')
            for x in res
        ]).reshape(-1)
        res = np.nan_to_num(res)
        return res

    def get_score(self, s1, s2, mode='cosine', is_rough=True):
        model = self.rough_w2v.model if is_rough else self.fine_w2v.model
        v1 = Embedding.wam(s1, model, agg='mean')
        v2 = Embedding.wam(s2, model, agg='mean')
        f_ssim = []
        if mode == 'cosine':
            f_ssim = scipy.spatial.distance.cdist(v1.reshape(1, -1), v2, 'cosine').reshape(-1)
        elif mode == 'euclidean':
            f_ssim = euclidean(v1, v2)
        elif mode == 'ts_ss':
            sim = Simcal(v1, v2)
            f_ssim = sim.Sector()
        f_ssim = [0.0 if math.isinf(x) or math.isnan(x) else x for x in f_ssim]
        return f_ssim


if __name__ == '__main__':
    cfg = get_cfg()
    ss = SemanticSimilarity(cfg)
    s1 = ['哈士奇', '拆家', '怎么办']
    s2 = ['狗狗', '吐', '了', '怎么办']
    print(ss.get_score(s1, s2))
    print(ss.get_score(['聪明'], ['智商']))
