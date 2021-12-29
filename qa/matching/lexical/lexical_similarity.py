from qa.matching import Matching
import Levenshtein
__all__ = ['LexicalSimilarity']


class LexicalSimilarity(Matching):
    __slot__ = []

    @staticmethod
    def levenshteinDistance(s1, s2):
        '''
        编辑距离——莱文斯坦距离,计算文本的相似度
        '''
        m = len(s1)
        n = len(s2)
        lensum = float(m + n)
        d = []
        for i in range(m + 1):
            d.append([i])
        del d[0][0]
        for j in range(n + 1):
            d[0].append(j)
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if s1[i - 1] == s2[j - 1]:
                    d[i].insert(j, d[i - 1][j - 1])
                else:
                    minimum = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                                  d[i - 1][j - 1] + 2)
                    d[i].insert(j, minimum)
        ldist = d[-1][-1]
        ratio = (lensum - ldist) / lensum
        #return {'distance':ldist, 'ratio':ratio}
        return ratio

    @staticmethod
    def lcs(text1, text2):
        M, N = len(text1), len(text2)
        dp = [[0] * (N + 1) for _ in range(M + 1)]
        p = 0
        max_len = 0
        for i in range(1, M + 1):
            for j in range(1, N + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        p = i + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return max_len, text1[p - max_len:p]

    @staticmethod
    def jaccard(s1, s2):
        s1 = set(s1)
        s2 = set(s2)
        ret1 = s1.intersection(s2)
        ret2 = s1.union(s2)
        jaccard = 1.0 * len(ret1) / len(ret2)

        return jaccard
    
    @staticmethod
    def levenshtein_multi(s1, s2_list):
        res = []
        for i in s2_list:
            res.append(Levenshtein.ratio(s1, i))
        return res

    @staticmethod
    def jaccard_multi(s1, s2_list):
        res = []
        for i in s2_list:
            res.append(LexicalSimilarity.jaccard(s1, i))
        return res

    def get_score(self, s1, s2, model='cosine'):
        if model == 'jaccard':
            sim = LexicalSimilarity.jaccard_multi(s1, s2)
        elif model == 'edit':
            sim = LexicalSimilarity.levenshtein_multi(s1, s2)
        elif model == 'lcs':
            sim, _ = LexicalSimilarity.lcs(s1, s2)
        return sim